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
# play_input <- in_w11 %>%
#   filter(game_id == 2023111906)  # replace with your actual play_id

play_output <- out_w11 %>%
  filter(play_id == 1180)  # replace with your actual play_id
# play_output <- out_w11 %>%
#   filter(game_id == 2023111906)  # replace with your actual play_id

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
# Based on dataset (play_input), useful additional features include:
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

# ------------------------------------------------------------
# visualize how off we are 
# -----------------------------------------------------------

# Compute prediction error for coloring
play_rmse <- play_output_with_pred %>%
  mutate(error = sqrt((x - x_pred)^2 + (y - y_pred)^2))

# Field layers only, not a full ggplot object
field_layers <- list(
  geom_rect(aes(xmin = 0, xmax = 120, ymin = 0, ymax = 53.3),
            fill = "palegreen4", color = "white"),
  geom_rect(aes(xmin = 0, xmax = 10, ymin = 0, ymax = 53.3), 
            fill = "gray30", alpha = 0.5),
  geom_rect(aes(xmin = 110, xmax = 120, ymin = 0, ymax = 53.3), 
            fill = "gray30", alpha = 0.5),
  geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linewidth = 0.5),
  scale_x_continuous(limits = c(0, 120)),
  scale_y_continuous(limits = c(0, 53.3)),
  coord_fixed(),
  theme_void()
)

# Plot trajectories
ggplot() +
  field_layers +
  # Actual trajectory
  geom_path(data = play_rmse, aes(x = x, y = y, group = nfl_id),
            color = "blue", linewidth = 1) +
  # Predicted trajectory
  geom_path(data = play_rmse, aes(x = x_pred, y = y_pred, group = nfl_id),
            color = "orange", linetype = "dashed", linewidth = 1) +
  # Current positions for last frame
  geom_point(data = play_rmse %>% filter(frame_id == max(frame_id)),
             aes(x = x_pred, y = y_pred, color = error), size = 3) +
  scale_color_gradient(low = "green", high = "red") +
  labs(title = paste("Predicted vs Actual Trajectories: Game", unique(play_rmse$game_id),
                     "Play", unique(play_rmse$play_id)),
       color = "Prediction Error (yds)") +
  theme(legend.position = "right")

# -------------------------------------
# rewrite compresses the entire past history into fixed-length summary 
# features that use all available past frames up to the current frame. 
# These summaries let the model “learn from all past frames” while 
# remaining a standard regression model.
# --------------------------------------

# compute slope of y ~ t for a numeric vector vec; returns 0 if length < 2
slope_by_time <- function(vec) {
  n <- length(vec)
  if (n < 2) return(0)
  t <- seq_len(n)
  m <- lm(vec ~ t)
  coef(m)[2]  # slope
}

# simple EWMA: alpha between 0 and 1; if empty return 0
ewma <- function(x, alpha = 0.5) {
  if (length(x) == 0) return(0)
  s <- x[1]
  for (i in 2:length(x)) s <- alpha * x[i] + (1 - alpha) * s
  s
}

# ---------------------------
# 0. Settings & incoming data assumed:
# play_input (with columns used earlier)
# play_output (contains true x,y for evaluation)
# ---------------------------
alpha_ewma <- 0.5  # decay for EWMA; tune as needed

# ---------------------------
# 1. Basic feature engineering (same as before)
# ---------------------------
play_features <- play_input %>%
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
# 2. Compute cumulative (past) summary features using all previous frames
#    For each frame t, we compute summaries over frames 1:(t-1)
# ---------------------------
play_features_with_past <- play_features %>%
  group_by(nfl_id, game_id, play_id) %>%
  arrange(frame_id) %>%
  do({
    df <- .
    n <- nrow(df)
    # prepare containers
    past_mean_x <- rep(NA_real_, n)
    past_mean_y <- rep(NA_real_, n)
    past_sd_x <- rep(NA_real_, n)
    past_sd_y <- rep(NA_real_, n)
    past_slope_x <- rep(NA_real_, n)
    past_slope_y <- rep(NA_real_, n)
    past_mean_vx <- rep(NA_real_, n)
    past_mean_vy <- rep(NA_real_, n)
    past_ewma_vx <- rep(NA_real_, n)
    past_ewma_vy <- rep(NA_real_, n)
    n_past <- rep(0L, n)
    
    for (i in seq_len(n)) {
      if (i == 1) {
        # no past frames
        past_mean_x[i] <- NA_real_
        past_mean_y[i] <- NA_real_
        past_sd_x[i] <- NA_real_
        past_sd_y[i] <- NA_real_
        past_slope_x[i] <- NA_real_
        past_slope_y[i] <- NA_real_
        past_mean_vx[i] <- NA_real_
        past_mean_vy[i] <- NA_real_
        past_ewma_vx[i] <- NA_real_
        past_ewma_vy[i] <- NA_real_
        n_past[i] <- 0L
      } else {
        idx <- seq_len(i - 1)  # past indices
        xs <- df$x[idx]
        ys <- df$y[idx]
        vxs <- df$vx[idx]
        vys <- df$vy[idx]
        past_mean_x[i]   <- mean(xs, na.rm = TRUE)
        past_mean_y[i]   <- mean(ys, na.rm = TRUE)
        past_sd_x[i]     <- ifelse(length(xs) > 1, sd(xs, na.rm = TRUE), 0)
        past_sd_y[i]     <- ifelse(length(ys) > 1, sd(ys, na.rm = TRUE), 0)
        past_slope_x[i]  <- slope_by_time(xs)
        past_slope_y[i]  <- slope_by_time(ys)
        past_mean_vx[i]  <- mean(vxs, na.rm = TRUE)
        past_mean_vy[i]  <- mean(vys, na.rm = TRUE)
        past_ewma_vx[i]  <- ewma(vxs, alpha = alpha_ewma)
        past_ewma_vy[i]  <- ewma(vys, alpha = alpha_ewma)
        n_past[i]        <- length(idx)
      }
    }
    
    df2 <- df %>%
      mutate(
        past_mean_x = past_mean_x,
        past_mean_y = past_mean_y,
        past_sd_x = past_sd_x,
        past_sd_y = past_sd_y,
        past_slope_x = past_slope_x,
        past_slope_y = past_slope_y,
        past_mean_vx = past_mean_vx,
        past_mean_vy = past_mean_vy,
        past_ewma_vx = past_ewma_vx,
        past_ewma_vy = past_ewma_vy,
        n_past = n_past
      )
    df2
  }) %>%
  ungroup()

# ---------------------------
# 3. Encode categorical variables as numeric indicators (keep as before)
# ---------------------------
to_predict <- play_features_with_past %>%
  filter(player_to_predict == TRUE) %>%
  mutate(
    play_direction_num = ifelse(play_direction == "right", 1, 0),
    player_position_WR = ifelse(player_position == "WR", 1, 0),
    player_position_RB = ifelse(player_position == "RB", 1, 0),
    player_position_QB = ifelse(player_position == "QB", 1, 0),
    player_role_Targeted = ifelse(player_role == "Targeted Receiver", 1, 0),
    player_role_Passer = ifelse(player_role == "Passer", 1, 0),
    player_role_Def = ifelse(player_role == "Defensive Coverage", 1, 0)
  )

players <- unique(to_predict$nfl_id)

# ---------------------------
# 4. Fit per-player linear models using summary-of-past features + current features
#    We keep a fallback approach: if a row has n_past == 0 or 1, we will later use
#    smaller models at prediction time (current-only or current+past1).
# ---------------------------
player_models <- map(players, function(pid) {
  df <- filter(to_predict, nfl_id == pid)
  # NOTE: use only rows where x_next is not NA (we filtered earlier), and exclude rows where model matrix has NAs in predictors
  df_fit <- df %>% filter(!is.na(past_mean_x) | n_past >= 0)  # keep all; we will rely on conditional predict later
  
  # Main model uses current + past-summary features
  lm_x <- lm(
    x_next ~ x + y + vx + vy + ax + ay +
      past_mean_x + past_mean_y + past_sd_x + past_sd_y + past_slope_x + past_slope_y +
      past_mean_vx + past_mean_vy + past_ewma_vx + past_ewma_vy + n_past +
      o + dir + play_direction_num +
      player_position_WR + player_position_RB + player_position_QB +
      player_role_Targeted + player_role_Passer + player_role_Def +
      ball_land_x + ball_land_y,
    data = df_fit
  )
  lm_y <- lm(
    y_next ~ x + y + vx + vy + ax + ay +
      past_mean_x + past_mean_y + past_sd_x + past_sd_y + past_slope_x + past_slope_y +
      past_mean_vx + past_mean_vy + past_ewma_vx + past_ewma_vy + n_past +
      o + dir + play_direction_num +
      player_position_WR + player_position_RB + player_position_QB +
      player_role_Targeted + player_role_Passer + player_role_Def +
      ball_land_x + ball_land_y,
    data = df_fit
  )
  list(x_model = lm_x, y_model = lm_y)
})
names(player_models) <- players

# ---------------------------
# 5. Conditional prediction using all-past summaries:
#    - if n_past == 0 : use current-only regression (fit on that player's rows where n_past==0? we'll fit a tiny model on all n_past==0 rows across player if needed)
#    - if n_past == 1 : use a model that includes past-1 summaries (we include them in the main model but some fields might be NA)
#    - else: use the full model above
# For simplicity we fit fallback models on-the-fly per player like before.
# ---------------------------
eval_df <- to_predict %>% mutate(x_pred = NA_real_, y_pred = NA_real_)

for (pid in unique(eval_df$nfl_id)) {
  df_player <- filter(eval_df, nfl_id == pid)
  for (play in unique(df_player$play_id)) {
    play_df <- filter(df_player, play_id == play) %>% arrange(frame_id)
    n <- nrow(play_df)
    if (n == 0) next
    
    # Frame-by-frame conditional predict:
    for (i in seq_len(n)) {
      row <- play_df[i, , drop = FALSE]
      npast <- row$n_past
      
      # choose model based on npast
      if (npast >= 2) {
        # use full per-player model
        mdl_x <- player_models[[as.character(pid)]]$x_model
        mdl_y <- player_models[[as.character(pid)]]$y_model
        pred_x <- predict(mdl_x, newdata = row)
        pred_y <- predict(mdl_y, newdata = row)
      } else if (npast == 1) {
        # fit a small temporary model for this player using rows with n_past >= 1 (so we get coefficient estimates)
        df_temp <- filter(df_player, n_past >= 1)
        # if not enough rows, fall back to global simple model
        if (nrow(df_temp) >= 3) {
          lm_x_1 <- lm(
            x_next ~ x + y + vx + vy + ax + ay +
              past_mean_x + past_mean_y + past_mean_vx + past_mean_vy +
              o + dir + play_direction_num +
              player_position_WR + player_position_RB + player_position_QB +
              player_role_Targeted + player_role_Passer + player_role_Def +
              ball_land_x + ball_land_y,
            data = df_temp
          )
          lm_y_1 <- lm(
            y_next ~ x + y + vx + vy + ax + ay +
              past_mean_x + past_mean_y + past_mean_vx + past_mean_vy +
              o + dir + play_direction_num +
              player_position_WR + player_position_RB + player_position_QB +
              player_role_Targeted + player_role_Passer + player_role_Def +
              ball_land_x + ball_land_y,
            data = df_temp
          )
          pred_x <- predict(lm_x_1, newdata = row)
          pred_y <- predict(lm_y_1, newdata = row)
        } else {
          # fallback global simple regression using current features only
          pred_x <- with(row, x + vx) # simple physics fallback (you can replace with small lm if desired)
          pred_y <- with(row, y + vy)
        }
      } else { # npast == 0
        # fallback: use current-only linear model (fit tiny per-player or global simple formula)
        # Try to fit per-player using only rows with n_past == 0 (rare). If not available, fallback to physics step.
        df_temp0 <- filter(df_player, n_past == 0)
        if (nrow(df_temp0) >= 3) {
          lm_x0 <- lm(x_next ~ x + y + vx + vy + ax + ay + o + dir +
                        play_direction_num + player_position_WR + player_position_RB + player_position_QB +
                        player_role_Targeted + player_role_Passer + player_role_Def +
                        ball_land_x + ball_land_y,
                      data = df_temp0)
          lm_y0 <- lm(y_next ~ x + y + vx + vy + ax + ay + o + dir +
                        play_direction_num + player_position_WR + player_position_RB + player_position_QB +
                        player_role_Targeted + player_role_Passer + player_role_Def +
                        ball_land_x + ball_land_y,
                      data = df_temp0)
          pred_x <- predict(lm_x0, newdata = row)
          pred_y <- predict(lm_y0, newdata = row)
        } else {
          # physics naive fallback: next position = current + velocity
          pred_x <- row$x + row$vx
          pred_y <- row$y + row$vy
        }
      }
      
      play_df$x_pred[i] <- as.numeric(pred_x)
      play_df$y_pred[i] <- as.numeric(pred_y)
    } # end frame loop
    
    # write back to eval_df
    eval_df <- eval_df %>%
      mutate(
        x_pred = ifelse(nfl_id == pid & play_id == play, play_df$x_pred, x_pred),
        y_pred = ifelse(nfl_id == pid & play_id == play, play_df$y_pred, y_pred)
      )
  } # end play loop
} # end pid loop

# ---------------------------
# 6. Evaluate model exactly like before
# ---------------------------
play_output_with_pred <- play_output %>%
  left_join(
    eval_df %>% select(nfl_id, frame_id, x_pred, y_pred),
    by = c("nfl_id", "frame_id")
  )

rmse_x <- rmse(play_output_with_pred$x, play_output_with_pred$x_pred)
rmse_y <- rmse(play_output_with_pred$y, play_output_with_pred$y_pred)
overall_rmse <- sqrt(mean((play_output_with_pred$x - play_output_with_pred$x_pred)^2 +
                            (play_output_with_pred$y - play_output_with_pred$y_pred)^2, na.rm = TRUE))

cat("RMSE x:", rmse_x, "\n")
cat("RMSE y:", rmse_y, "\n")
cat("Overall RMSE:", overall_rmse, "\n")

