# ====================================================
# NOTES AFTER 11/3 MTG 
# something is wrong in our preprocessing of data
# think more carefully about variables putting into the model
# 1 player, 1 play no interaction
# (can't have more covariates than data points)
# see which 5 covariates are the highest
# (derivative in placement and location)
# s, a, v, direction
# 
# 2 players, 1 play and see their interaction
# 3,4,5 ....
# ====================================================

# this code right now works with the entire game or play id you specifiy
# jenna and anyone else can edit this to fit 1 player to we can see how to better preprocess

library(tidyverse)
library(tidymodels)
library(embed)        # for step_lencode_mixed()
library(slider)       # for rolling lag features
library(Metrics)      # for rmse()
library(ranger)       # fast RF backend
library(tidyverse)
library(tidymodels)
library(vroom)

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

# ====================================================
# 0. Select play or game
# ====================================================
# game_id_to_use <- 2023111906
game_id_to_use <- 1180
play_id <- 1180

#make smaller csv for input and output

# Uncomment whichever you want
play_input <- in_w11 %>%
#   filter(game_id == game_id_to_use)
  filter(play_id == 1180)

# singlegame_input <- vroom_write(play_input, "game_input.csv")

play_output <- out_w11 %>%
  # filter(game_id == game_id_to_use)
  filter(play_id == 1180)

# singlegame_output <- vroom_write(play_output, "game_output.csv")

# ====================================================
# 1. Feature engineering on input
# ====================================================
play_features <- play_input %>%
  arrange(nfl_id, frame_id) %>%
  # compute velocities and accelerations
  mutate(
    dir_rad = dir * pi / 180,
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    # directionality & angular changes
    speed_sq = s^2,
    acc_sq = a^2,
    vx_vy_angle = atan2(vy, vx),
    dir_change = dir - lag(dir),
    orientation_cos = cos(o * pi/180),
    orientation_sin = sin(o * pi/180),
    dir_cos = cos(dir * pi/180),
    dir_sin = sin(dir * pi/180)
  ) %>%
  group_by(nfl_id) %>%
  mutate(
    x_lag1 = lag(x, 1),
    y_lag1 = lag(y, 1),
    vx_lag1 = lag(vx, 1),
    vy_lag1 = lag(vy, 1),
    ax_lag1 = lag(ax, 1),
    ay_lag1 = lag(ay, 1),
    n_past = row_number() - 1
  ) %>%
  ungroup() %>%
  filter(player_to_predict == TRUE)
# change this to use all players instead of only predictors
# 5 lag might be better for lag

play_features <- play_features %>%
  group_by(nfl_id, game_id, play_id) %>%
  arrange(frame_id) %>%
  mutate(
    x_next = lead(x, 1),
    y_next = lead(y, 1)
  ) %>%
  ungroup()

# ====================================================
# 2. Prepare model matrix
# ====================================================
# Convert categorical variables to dummies
categorical_vars <- c("play_direction", "player_position", "player_role")
dummies <- model.matrix(~ . - 1, data = play_features[categorical_vars]) %>% as.data.frame()

# Combine numeric + dummy predictors
predictors <- play_features %>%
  select(vx, vy, ax, ay, s, a, o, dir,
         speed_sq, acc_sq, vx_vy_angle, dir_change,
         orientation_cos, orientation_sin, dir_cos, dir_sin,
         x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past) %>%
  bind_cols(dummies)

fit_data <- bind_cols(
  predictors,
  play_features[, c("x_next", "y_next")]
) %>%
  drop_na()  # remove rows where x_next or y_next are NA

# TO DOUBLE CHECK IF MODEL IS PREDICTING WELL -----------------------
# plot(cbind((fit_data$x_next,fit_data$y_next))
#      # what the model is actually doing 
#      points(mv_lm$fitted.values, col='red')

# 3. Fit multivariate linear model -----------------------------------
# mv_lm <- lm(cbind(x_next, y_next) ~ ., data = bind_cols(predictors, play_features[, c("x_next", "y_next")]))
mv_lm <- lm(cbind(x_next, y_next) ~ ., data = fit_data)

# 4. Generate features for output frames--------------------------------
# Join the same features to play_output
play_test <- play_output %>%
  left_join(
    play_features %>%
      select(game_id, play_id, nfl_id, frame_id, vx, vy, ax, ay, s, a, o, dir,
             speed_sq, acc_sq, vx_vy_angle, dir_change,
             orientation_cos, orientation_sin, dir_cos, dir_sin,
             x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past,
             play_direction, player_position, player_role),
    by = c("game_id","play_id","nfl_id","frame_id")
  )

# Build dummy matrix for test set
dummies_test <- model.matrix(~ . - 1, data = play_test[categorical_vars]) %>% as.data.frame()
test_predictors <- play_test %>%
  select(vx, vy, ax, ay, s, a, o, dir,
         speed_sq, acc_sq, vx_vy_angle, dir_change,
         orientation_cos, orientation_sin, dir_cos, dir_sin,
         x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past) %>%
  bind_cols(dummies_test)

# ====================================================
# 5. Predict x and y for output frames
# ====================================================
preds <- predict(mv_lm, newdata = test_predictors)
play_output_preds <- play_test %>%
  bind_cols(as.data.frame(preds)) %>%
  rename(x_pred = x_next, y_pred = y_next)

# ====================================================
# 6. Optional: Evaluate RMSE if x,y exist in output
# ====================================================
if(all(c("x","y") %in% colnames(play_output_preds))){
  rmse_x <- sqrt(mean((play_output_preds$x - play_output_preds$x_pred)^2))
  rmse_y <- sqrt(mean((play_output_preds$y - play_output_preds$y_pred)^2))
  overall_rmse <- sqrt(mean((play_output_preds$x - play_output_preds$x_pred)^2 +
                              (play_output_preds$y - play_output_preds$y_pred)^2))
  
  cat("=== Multivariate LM ===\n")
  cat("RMSE x:", rmse_x, "\n")
  cat("RMSE y:", rmse_y, "\n")
  cat("Overall RMSE:", overall_rmse, "\n")
}



diff_x <- play_output_preds$x - play_output_preds$x_pred
diff_y <- play_output_preds$y - play_output_preds$y_pred

# for unique play in play_id
# plot(x=diff_x, y=diff_y)

# Assuming your data is in a data frame called 'data_df' 
# and contains columns 'play_id', 'x', and 'y'.

# 1. Get all unique play IDs
unique_plays <- unique(play_output_preds$play_id)

# 2. Loop through each unique play ID
for (current_play_id in unique_plays) {
  
  # Filter the data frame to get only points for the current play ID
  play_data <- subset(play_output_preds, play_id == current_play_id)
  
  # Create a plot for this specific play
  plot(x = play_data$x, y = play_data$y,
       main = paste("Play ID:", current_play_id), # Set the plot title
       xlab = "X Coordinate",
       ylab = "Y Coordinate",
       type = "b",   # 'b' plots both points and lines
       pch = 19,     # Solid points
       col = "blue",
       # Set consistent axis limits if you want comparable plots:
       xlim = c(min(play_output_preds$x, na.rm = TRUE), max(play_output_preds$x, na.rm = TRUE)),
       ylim = c(min(play_output_preds$y, na.rm = TRUE), max(play_output_preds$y, na.rm = TRUE))
  )
  
  # Optional: If you want R to pause between plots for viewing:
  # locator(1) 
}

