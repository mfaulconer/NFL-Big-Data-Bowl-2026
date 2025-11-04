# now we are doing 1 full game 

library(vroom)
library(dplyr)
library(ggplot2)
library(gganimate)
library(dplyr)
library(Metrics)
library(purrr)
library(randomForest)
library(tidyr)

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

# ---------------------------------------------------
# 1. Select one game
# ---------------------------------------------------
game_id_to_use <- 2023111906

play_input <- in_w11 %>%
  filter(game_id == game_id_to_use) %>%
  arrange(nfl_id, play_id, frame_id)

play_output <- out_w11 %>%
  filter(game_id == game_id_to_use) %>%
  arrange(nfl_id, play_id, frame_id)

# ---------------------------------------------------
# 2. Feature engineering on INPUT (training data)
# ---------------------------------------------------
play_features <- play_input %>%
  group_by(nfl_id, play_id) %>%
  mutate(
    dir_rad = dir * pi / 180,
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    x_next = lead(x, 1),
    y_next = lead(y, 1)
  ) %>%
  ungroup() %>%
  filter(!is.na(x_next), !is.na(y_next)) %>%
  mutate(
    play_direction_num = ifelse(play_direction == "right", 1, 0),
    player_position_WR = ifelse(player_position == "WR", 1, 0),
    player_position_RB = ifelse(player_position == "RB", 1, 0),
    player_position_QB = ifelse(player_position == "QB", 1, 0),
    player_role_Targeted = ifelse(player_role == "Targeted Receiver", 1, 0),
    player_role_Passer = ifelse(player_role == "Passer", 1, 0),
    player_role_Def = ifelse(player_role == "Defensive Coverage", 1, 0)
  )

# ---------------------------------------------------
# 3. Predict per-play, per-frame using dynamic lag features
# ---------------------------------------------------
eval_df <- play_output %>%
  mutate(x_pred_lm = NA_real_, y_pred_lm = NA_real_,
         x_pred_rf = NA_real_, y_pred_rf = NA_real_)

players <- unique(play_output$nfl_id)

for(pid in players){
  df_input_player <- filter(play_features, nfl_id == pid)
  df_output_player <- filter(play_output, nfl_id == pid)
  
  plays <- unique(df_input_player$play_id)
  
  for(play in plays){
    input_play <- filter(df_input_player, play_id == play)
    output_play <- filter(df_output_player, play_id == play)
    n_frames <- nrow(output_play)
    
    # Store predictions
    x_preds_lm <- numeric(n_frames)
    y_preds_lm <- numeric(n_frames)
    x_preds_rf <- numeric(n_frames)
    y_preds_rf <- numeric(n_frames)
    
    for(f in 1:n_frames){
      # Select all available previous frames for this frame
      if(f == 1){
        # Frame 1: no lag
        df_train <- input_play[1, ]
        pred_formula <- as.formula(
          paste("x_next ~ x + y + vx + vy + ax + ay + o + dir + play_direction_num +",
                "player_position_WR + player_position_RB + player_position_QB +",
                "player_role_Targeted + player_role_Passer + player_role_Def +",
                "ball_land_x + ball_land_y")
        )
      } else {
        # Frames >1: include all available lagged features
        lag_indices <- 1:(f-1)
        lag_features <- paste0("x_lag", lag_indices, collapse = " + ")
        lag_features <- paste(lag_features, paste0("y_lag", lag_indices, collapse = " + "), sep = " + ")
        lag_features <- paste(lag_features,
                              paste0("vx_lag", lag_indices, collapse = " + "), sep = " + ")
        lag_features <- paste(lag_features,
                              paste0("vy_lag", lag_indices, collapse = " + "), sep = " + ")
        lag_features <- paste(lag_features,
                              paste0("ax_lag", lag_indices, collapse = " + "), sep = " + ")
        lag_features <- paste(lag_features,
                              paste0("ay_lag", lag_indices, collapse = " + "), sep = " + ")
        
        base_features <- "x + y + vx + vy + ax + ay + o + dir + play_direction_num + \
player_position_WR + player_position_RB + player_position_QB + \
player_role_Targeted + player_role_Passer + player_role_Def + ball_land_x + ball_land_y"
        
        pred_formula <- as.formula(paste("x_next ~", paste(base_features, lag_features, sep = " + ")))
      }
      
      # Linear model prediction
      lm_x <- lm(pred_formula, data = input_play[1:f, , drop=FALSE])
      x_preds_lm[f] <- predict(lm_x, newdata = input_play[f, , drop=FALSE])
      
      lm_y <- update(pred_formula, y_next ~ .)
      y_preds_lm[f] <- predict(lm_y, newdata = input_play[f, , drop=FALSE])
      
      # Random Forest prediction
      rf_x <- randomForest(pred_formula, data = input_play[1:f, , drop=FALSE], ntree = 100, mtry = 6)
      x_preds_rf[f] <- predict(rf_x, newdata = input_play[f, , drop=FALSE])
      
      rf_y <- randomForest(update(pred_formula, y_next ~ .), data = input_play[1:f, , drop=FALSE], ntree = 100, mtry = 6)
      y_preds_rf[f] <- predict(rf_y, newdata = input_play[f, , drop=FALSE])
    }
    
    # Save predictions into eval_df
    eval_df <- eval_df %>%
      mutate(
        x_pred_lm = ifelse(nfl_id == pid & play_id == play, x_preds_lm, x_pred_lm),
        y_pred_lm = ifelse(nfl_id == pid & play_id == play, y_preds_lm, y_pred_lm),
        x_pred_rf = ifelse(nfl_id == pid & play_id == play, x_preds_rf, x_pred_rf),
        y_pred_rf = ifelse(nfl_id == pid & play_id == play, y_preds_rf, y_pred_rf)
      )
  }
}

# ---------------------------------------------------
# 4. Evaluate predictions
# ---------------------------------------------------
rmse_x_lm <- rmse(eval_df$x, eval_df$x_pred_lm)
rmse_y_lm <- rmse(eval_df$y, eval_df$y_pred_lm)
overall_rmse_lm <- sqrt(mean((eval_df$x - eval_df$x_pred_lm)^2 + (eval_df$y - eval_df$y_pred_lm)^2))

rmse_x_rf <- rmse(eval_df$x, eval_df$x_pred_rf)
rmse_y_rf <- rmse(eval_df$y, eval_df$y_pred_rf)
overall_rmse_rf <- sqrt(mean((eval_df$x - eval_df$x_pred_rf)^2 + (eval_df$y - eval_df$y_pred_rf)^2))

cat("=== Linear Auto-Regressive Model ===\n")
cat("RMSE x:", rmse_x_lm, "\n")
cat("RMSE y:", rmse_y_lm, "\n")
cat("Overall RMSE:", overall_rmse_lm, "\n\n")

cat("=== Random Forest Model ===\n")
cat("RMSE x:", rmse_x_rf, "\n")
cat("RMSE y:", rmse_y_rf, "\n")
cat("Overall RMSE:", overall_rmse_rf, "\n")

