# multivariate regression on just 1 play

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

library(dplyr)
library(purrr)
library(tidyr)

# ---------------------------
# Helper functions
# ---------------------------
slope_by_time <- function(vec) {
  n <- length(vec)
  if (n < 2) return(0)
  t <- seq_len(n)
  coef(lm(vec ~ t))[2]
}

ewma <- function(x, alpha = 0.5) {
  if (length(x) == 0) return(0)
  s <- x[1]
  for (i in 2:length(x)) s <- alpha * x[i] + (1 - alpha) * s
  s
}

# ---------------------------
# 1. Prepare input features
# ---------------------------
play_input <- in_w11 %>%
  filter(play_id == 1180) %>%
  arrange(nfl_id, game_id, play_id, frame_id) %>%
  group_by(nfl_id, game_id, play_id) %>%
  mutate(
    dir_rad = dir * pi / 180,
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    x_next = lead(x, 1),
    y_next = lead(y, 1)
  ) %>%
  ungroup()

# ---------------------------
# 2. Compute cumulative summaries (learn from *all* past frames)
# ---------------------------
alpha_ewma <- 0.5

play_features <- play_input %>%
  group_by(nfl_id, game_id, play_id) %>%
  arrange(frame_id) %>%
  mutate(
    past_mean_x = map_dbl(seq_along(x), ~ if (.x == 1) NA else mean(x[1:(.x - 1)], na.rm = TRUE)),
    past_mean_y = map_dbl(seq_along(y), ~ if (.x == 1) NA else mean(y[1:(.x - 1)], na.rm = TRUE)),
    past_sd_x   = map_dbl(seq_along(x), ~ if (.x == 1) 0 else sd(x[1:(.x - 1)], na.rm = TRUE)),
    past_sd_y   = map_dbl(seq_along(y), ~ if (.x == 1) 0 else sd(y[1:(.x - 1)], na.rm = TRUE)),
    past_slope_x = map_dbl(seq_along(x), ~ slope_by_time(x[1:(.x - 1)])),
    past_slope_y = map_dbl(seq_along(y), ~ slope_by_time(y[1:(.x - 1)])),
    past_mean_vx = map_dbl(seq_along(vx), ~ if (.x == 1) 0 else mean(vx[1:(.x - 1)], na.rm = TRUE)),
    past_mean_vy = map_dbl(seq_along(vy), ~ if (.x == 1) 0 else mean(vy[1:(.x - 1)], na.rm = TRUE)),
    past_ewma_vx = map_dbl(seq_along(vx), ~ ewma(vx[1:(.x - 1)], alpha_ewma)),
    past_ewma_vy = map_dbl(seq_along(vy), ~ ewma(vy[1:(.x - 1)], alpha_ewma)),
    n_past = frame_id - min(frame_id)
  ) %>%
  ungroup()

# ---------------------------
# 3. Encode categorical features
# ---------------------------
# play_features <- play_features %>%
#   mutate(
#     play_direction_num = ifelse(play_direction == "right", 1, 0),
#     player_position_WR = ifelse(player_position == "WR", 1, 0),
#     player_position_RB = ifelse(player_position == "RB", 1, 0),
#     player_position_QB = ifelse(player_position == "QB", 1, 0),
#     player_role_Targeted = ifelse(player_role == "Targeted Receiver", 1, 0),
#     player_role_Passer = ifelse(player_role == "Passer", 1, 0),
#     player_role_Def = ifelse(player_role == "Defensive Coverage", 1, 0)
#   )
# Feature engineering with tidymodels
play_recipe <- recipe(x_next + y_next ~ ., data = play_features) %>%
  step_mutate(
    play_direction = as.factor(play_direction),
    player_position = as.factor(player_position),
    player_role = as.factor(player_role)
  ) %>%
  step_lencode_mixed(player_position, outcome = vars(x_next, y_next)) %>%
  step_lencode_mixed(player_role, outcome = vars(x_next, y_next)) %>% # target encoding
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  # step_rm(nfl_id, game_id, play_id, frame_id) # why would we remove this? 

play_prep <- prep(play_recipe)
play_ready <- bake(play_prep, new_data = NULL)


# ---------------------------
# 4. Fit multivariate regression per player
# ---------------------------
players <- unique(play_features$nfl_id)
player_models <- map(players, function(pid) {
  df <- filter(play_features, nfl_id == pid, !is.na(x_next), !is.na(y_next))
  
  # Drop unused columns to prevent NA issues
  df_fit <- df %>% select(
    x_next, y_next, x, y, vx, vy, ax, ay,
    past_mean_x, past_mean_y, past_sd_x, past_sd_y,
    past_slope_x, past_slope_y,
    past_mean_vx, past_mean_vy, past_ewma_vx, past_ewma_vy, n_past,
    o, dir, play_direction_num,
    player_position_WR, player_position_RB, player_position_QB,
    player_role_Targeted, player_role_Passer, player_role_Def,
    ball_land_x, ball_land_y
  )
  
  # Multivariate model
  mv_fit <- lm(
    cbind(x_next, y_next) ~ x + y + vx + vy + ax + ay +
      past_mean_x + past_mean_y + past_sd_x + past_sd_y +
      past_slope_x + past_slope_y + past_mean_vx + past_mean_vy +
      past_ewma_vx + past_ewma_vy + n_past +
      o + dir + play_direction_num +
      player_position_WR + player_position_RB + player_position_QB +
      player_role_Targeted + player_role_Passer + player_role_Def +
      ball_land_x + ball_land_y,
    data = df_fit
  )
  mv_fit
})
names(player_models) <- players

# ---------------------------
# 5. Predict next-frame positions for each frame
# ---------------------------
predictions <- play_features %>%
  mutate(x_pred = NA_real_, y_pred = NA_real_)

for (pid in players) {
  mdl <- player_models[[as.character(pid)]]
  df_player <- filter(play_features, nfl_id == pid)
  
  pred_mat <- try(predict(mdl, newdata = df_player), silent = TRUE)
  
  if (!inherits(pred_mat, "try-error")) {
    predictions <- predictions %>%
      mutate(
        x_pred = ifelse(nfl_id == pid, pred_mat[, 1], x_pred),
        y_pred = ifelse(nfl_id == pid, pred_mat[, 2], y_pred)
      )
  } else {
    # fallback for model failure
    predictions <- predictions %>%
      mutate(
        x_pred = ifelse(nfl_id == pid, x + vx, x_pred),
        y_pred = ifelse(nfl_id == pid, y + vy, y_pred)
      )
  }
}

# ---------------------------
# 6. Join with output and evaluate
# ---------------------------
play_output <- out_w11 %>% filter(play_id == 1180)

eval_df <- play_output %>%
  left_join(predictions %>% select(nfl_id, frame_id, x_pred, y_pred), 
            by = c("nfl_id", "frame_id"))

rmse <- function(a, b) sqrt(mean((a - b)^2, na.rm = TRUE))

rmse_x <- rmse(eval_df$x, eval_df$x_pred)
rmse_y <- rmse(eval_df$y, eval_df$y_pred)
overall_rmse <- sqrt(mean((eval_df$x - eval_df$x_pred)^2 + (eval_df$y - eval_df$y_pred)^2, na.rm = TRUE))

cat("=== Multivariate Regression (Full-Past Summary) ===\n")
cat("RMSE x:", rmse_x, "\n")
cat("RMSE y:", rmse_y, "\n")
cat("Overall RMSE:", overall_rmse, "\n")
