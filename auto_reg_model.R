# auto regressive linear model

library(vroom)
library(dplyr)
library(ggplot2)
library(gganimate)
library(dplyr)
library(Metrics)
library(purrr)

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

# --------------------------
# 1 Prepare features
# --------------------------
play_input <- in_w11 %>%
  filter(play_id == 1180)  # replace with your actual play_id

play_output <- out_w11 %>%
  filter(play_id == 1180)  # replace with your actual play_id

play_features <- play_input %>%
  arrange(nfl_id, game_id, play_id, frame_id) %>%
  group_by(nfl_id, game_id, play_id) %>%
  mutate(
    # Convert direction from degrees to radians
    dir_rad = dir * pi / 180,
    # Velocity components
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    # Acceleration components
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    # Next frame positions (target)
    x_next = lead(x, 1),
    y_next = lead(y, 1)
  ) %>%
  ungroup() %>%
  filter(!is.na(x_next), !is.na(y_next))  # drop last frame which has no next position

# --------------------------
# 2 Filter players to predict
# --------------------------

to_predict <- play_features %>% filter(player_to_predict == TRUE)

# --------------------------
# 3 Fit linear AR models per player
# --------------------------

players <- unique(to_predict$nfl_id)

player_models <- map(players, function(pid){
  df <- filter(to_predict, nfl_id == pid)
  
  # Linear regression for x_next
  lm_x <- lm(x_next ~ x + vx + ax, data = df)
  
  # Linear regression for y_next
  lm_y <- lm(y_next ~ y + vy + ay, data = df)
  
  list(x_model = lm_x, y_model = lm_y)
})
names(player_models) <- players

# --------------------------
# 4 Make predictions
# --------------------------

predictions <- to_predict %>%
  rowwise() %>%
  mutate(
    x_pred = predict(player_models[[as.character(nfl_id)]]$x_model, newdata = cur_data()),
    y_pred = predict(player_models[[as.character(nfl_id)]]$y_model, newdata = cur_data())
  ) %>%
  ungroup()

# --------------------------
# 5 Merge with actual output
# --------------------------

# Add predictions to play_output without changing existing columns
play_output_with_pred <- play_output %>%
  left_join(
    predictions %>% select(nfl_id, frame_id, x_pred, y_pred),
    by = c("nfl_id", "frame_id")
  )

eval_df <- play_output_with_pred

rmse_x <- rmse(eval_df$x, eval_df$x_pred)
rmse_y <- rmse(eval_df$y, eval_df$y_pred)
overall_rmse <- sqrt(mean((eval_df$x - eval_df$x_pred)^2 + (eval_df$y - eval_df$y_pred)^2))

cat("RMSE x:", rmse_x, "\n")
cat("RMSE y:", rmse_y, "\n")
cat("Overall RMSE:", overall_rmse, "\n")


# -----------------------------------------
# 
# Based on your dataset (play_input), useful additional features include:
#   Player orientation & direction: o, dir
# Player role/position: player_role, player_position (categorical)
# Play context: play_direction
# Ball landing position: ball_land_x, ball_land_y
# Previous velocities/accelerations: vx, vy, ax, ay (already included)
# We will also include lagged features from previous frames (t-1, t-2) to capture motion history.
# --------------------------------------------

n_lags <- 2  # number of previous frames to include

# --- 1. Feature engineering ---
play_features <- play_input %>%
  arrange(nfl_id, game_id, play_id, frame_id) %>%
  group_by(nfl_id, game_id, play_id) %>%
  mutate(
    dir_rad = dir * pi / 180,
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    
    # Lagged features
    x_lag1 = lag(x, 1), y_lag1 = lag(y, 1),
    vx_lag1 = lag(vx, 1), vy_lag1 = lag(vy, 1),
    ax_lag1 = lag(ax, 1), ay_lag1 = lag(ay, 1),
    
    x_lag2 = lag(x, 2), y_lag2 = lag(y, 2),
    vx_lag2 = lag(vx, 2), vy_lag2 = lag(vy, 2),
    ax_lag2 = lag(ax, 2), ay_lag2 = lag(ay, 2),
    
    # Targets
    x_next = lead(x, 1),
    y_next = lead(y, 1)
  ) %>%
  ungroup() %>%
  filter(!is.na(x_next)) %>%  # remove rows where next frame is NA
  mutate(
    player_role = as.factor(player_role),
    player_position = as.factor(player_position),
    play_direction = as.factor(play_direction)
  )

# --- 2. Prepare data for prediction ---
to_predict <- play_features %>%
  filter(player_to_predict == TRUE) %>%
  mutate(
    # one-hot encode categorical variables
    play_direction_num = ifelse(play_direction == "right", 1, 0),
    player_position_WR = ifelse(player_position == "WR", 1, 0),
    player_position_RB = ifelse(player_position == "RB", 1, 0),
    player_position_QB = ifelse(player_position == "QB", 1, 0),
    player_role_Targeted = ifelse(player_role == "Targeted Receiver", 1, 0),
    player_role_Passer = ifelse(player_role == "Passer", 1, 0),
    player_role_Def = ifelse(player_role == "Defensive Coverage", 1, 0)
  )

players <- unique(to_predict$nfl_id)

# --- 3. Fit per-player linear models ---
player_models <- map(players, function(pid){
  df <- filter(to_predict, nfl_id == pid)
  
  lm_x <- lm(
    x_next ~ x + y + vx + vy + ax + ay +
      x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
      x_lag2 + y_lag2 + vx_lag2 + vy_lag2 + ax_lag2 + ay_lag2 +
      o + dir + play_direction_num +
      player_position_WR + player_position_RB + player_position_QB +
      player_role_Targeted + player_role_Passer + player_role_Def +
      ball_land_x + ball_land_y,
    data = df
  )
  
  lm_y <- lm(
    y_next ~ x + y + vx + vy + ax + ay +
      x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
      x_lag2 + y_lag2 + vx_lag2 + vy_lag2 + ax_lag2 + ay_lag2 +
      o + dir + play_direction_num +
      player_position_WR + player_position_RB + player_position_QB +
      player_role_Targeted + player_role_Passer + player_role_Def +
      ball_land_x + ball_land_y,
    data = df
  )
  
  list(x_model = lm_x, y_model = lm_y)
})
names(player_models) <- players

# --- 4. Conditional lag prediction ---
eval_df <- to_predict %>%
  mutate(x_pred = NA_real_, y_pred = NA_real_)

for(pid in unique(eval_df$nfl_id)){
  df_player <- filter(eval_df, nfl_id == pid)
  
  for(play in unique(df_player$play_id)){
    play_df <- filter(df_player, play_id == play)
    n <- nrow(play_df)
    
    # Frame 1: no lag
    if(n >= 1){
      lm_first_x <- lm(
        x_next ~ x + y + vx + vy + ax + ay + o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df[1, ]
      )
      play_df$x_pred[1] <- predict(lm_first_x, newdata = play_df[1, ])
      
      lm_first_y <- lm(
        y_next ~ x + y + vx + vy + ax + ay + o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df[1, ]
      )
      play_df$y_pred[1] <- predict(lm_first_y, newdata = play_df[1, ])
    }
    
    # Frame 2: only lag1
    if(n >= 2){
      lm_second_x <- lm(
        x_next ~ x + y + vx + vy + ax + ay +
          x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
          o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df[1:2, ]
      )
      play_df$x_pred[2] <- predict(lm_second_x, newdata = play_df[2, ])
      
      lm_second_y <- lm(
        y_next ~ x + y + vx + vy + ax + ay +
          x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
          o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df[1:2, ]
      )
      play_df$y_pred[2] <- predict(lm_second_y, newdata = play_df[2, ])
    }
    
    # Frame 3+: full lags
    if(n > 2){
      lm_full_x <- lm(
        x_next ~ x + y + vx + vy + ax + ay +
          x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
          x_lag2 + y_lag2 + vx_lag2 + vy_lag2 + ax_lag2 + ay_lag2 +
          o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df
      )
      lm_full_y <- lm(
        y_next ~ x + y + vx + vy + ax + ay +
          x_lag1 + y_lag1 + vx_lag1 + vy_lag1 + ax_lag1 + ay_lag1 +
          x_lag2 + y_lag2 + vx_lag2 + vy_lag2 + ax_lag2 + ay_lag2 +
          o + dir +
          player_position_WR + player_position_RB + player_position_QB +
          player_role_Targeted + player_role_Passer + player_role_Def +
          play_direction_num + ball_land_x + ball_land_y,
        data = play_df
      )
      play_df$x_pred[3:n] <- predict(lm_full_x, newdata = play_df[3:n, ])
      play_df$y_pred[3:n] <- predict(lm_full_y, newdata = play_df[3:n, ])
    }
    
    # Update eval_df
    eval_df <- eval_df %>%
      mutate(
        x_pred = ifelse(nfl_id == pid & play_id == play, play_df$x_pred, x_pred),
        y_pred = ifelse(nfl_id == pid & play_id == play, play_df$y_pred, y_pred)
      )
  }
}

# --- 5. Evaluate model ---
play_output_with_pred <- play_output %>%
  left_join(
    eval_df %>% select(nfl_id, frame_id, x_pred, y_pred),
    by = c("nfl_id", "frame_id")
  )

rmse_x <- rmse(play_output_with_pred$x, play_output_with_pred$x_pred)
rmse_y <- rmse(play_output_with_pred$y, play_output_with_pred$y_pred)
overall_rmse <- sqrt(mean((play_output_with_pred$x - play_output_with_pred$x_pred)^2 +
                            (play_output_with_pred$y - play_output_with_pred$y_pred)^2))

cat("RMSE x:", rmse_x, "\n")
cat("RMSE y:", rmse_y, "\n")
cat("Overall RMSE:", overall_rmse, "\n")

