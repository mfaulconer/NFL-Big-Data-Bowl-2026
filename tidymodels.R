# this compared linear regression to random forests

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
game_id_to_use <- 2023111906
play_input <- in_w11 %>%
  filter(game_id == game_id_to_use)
  # filter(play_id == 1180)

play_output <- out_w11 %>%
  filter(game_id == game_id_to_use)
  # filter(play_id == 1180)

# ====================================================
# 1. Feature engineering on input
# ====================================================
play_features <- play_input %>%
  arrange(nfl_id, frame_id) %>%
  mutate(
    dir_rad = dir * pi / 180,
    vx = s * cos(dir_rad),
    vy = s * sin(dir_rad),
    ax = a * cos(dir_rad),
    ay = a * sin(dir_rad),
    speed_sq = s^2,
    acc_sq = a^2,
    vx_vy_angle = atan2(vy, vx),
    dir_change = dir - lag(dir, default = dir[1]),
    orientation_cos = cos(o * pi / 180),
    orientation_sin = sin(o * pi / 180),
    dir_cos = cos(dir * pi / 180),
    dir_sin = sin(dir * pi / 180)
  ) %>%
  group_by(nfl_id) %>%
  mutate(
    x_lag1 = lag(x, 1, default = x[1]),
    y_lag1 = lag(y, 1, default = y[1]),
    vx_lag1 = lag(vx, 1, default = vx[1]),
    vy_lag1 = lag(vy, 1, default = vy[1]),
    ax_lag1 = lag(ax, 1, default = ax[1]),
    ay_lag1 = lag(ay, 1, default = ay[1]),
    n_past = row_number() - 1
  ) %>%
  ungroup()

# ====================================================
# 2. Create recipe
# ====================================================
play_recipe <- recipe(cbind(x_next, y_next) ~ ., data = play_features) %>%
  update_role(game_id, play_id, frame_id, nfl_id, new_role = "id") %>%
  step_mutate(
    play_direction = as.factor(play_direction),
    player_position = as.factor(player_position),
    player_role = as.factor(player_role)
  ) %>%
  step_dummy(all_nominal_predictors()) %>%
  # interactions
  step_interact(~ (vx + vy + ax + ay + s + a + o + dir + speed_sq + acc_sq + vx_vy_angle + dir_change + orientation_cos + orientation_sin + dir_cos + dir_sin) : starts_with("player_position_")) %>%
  step_interact(~ (vx + vy + ax + ay + s + a + o + dir + speed_sq + acc_sq + vx_vy_angle + dir_change + orientation_cos + orientation_sin + dir_cos + dir_sin) : starts_with("player_role_")) %>%
  step_interact(~ (vx + vy + s + speed_sq + vx_vy_angle) : starts_with("play_direction_")) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

play_prep <- prep(play_recipe)
train_ready <- bake(play_prep, new_data = NULL)

# ====================================================
# 3. Model specifications
# ====================================================
lm_spec <- linear_reg() %>% set_engine("lm")
rf_spec <- rand_forest(mtry = 12, trees = 200, min_n = 5) %>% set_engine("ranger") %>% set_mode("regression")

# ====================================================
# 4. Build workflows
# ====================================================
lm_wf <- workflow() %>%
  add_model(lm_spec) %>%
  add_recipe(play_recipe)

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(play_recipe)

# ====================================================
# 5. Fit models on input data
# ====================================================
lm_fit <- fit(lm_wf, data = play_features)
rf_fit <- fit(rf_wf, data = play_features)

# ====================================================
# 6. Generate features for output frames

play_features <- play_features %>%
  filter(player_to_predict == TRUE)

play_test <- play_output %>%
  filter(nfl_id %in% play_features$nfl_id) %>%  # only players we want to predict
  left_join(
    play_features %>%
      select(game_id, play_id, nfl_id, frame_id, x, y, vx, vy, ax, ay, s, a, dir, o,
             player_position, player_role,
             x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past,
             speed_sq, acc_sq, vx_vy_angle, dir_change, orientation_cos, orientation_sin, dir_cos, dir_sin),
    by = c("game_id","play_id","nfl_id","frame_id")
  )

# ====================================================
# 7. Bake output features
# ====================================================
play_test_ready <- bake(play_prep, new_data = play_test)

# ====================================================
# 8. Predict x and y
# ====================================================
play_test_preds <- play_test %>%
  bind_cols(
    x_pred_lm = predict(lm_fit, new_data = play_test_ready)$.pred,
    y_pred_lm = predict(lm_fit, new_data = play_test_ready)$.pred,
    x_pred_rf = predict(rf_fit, new_data = play_test_ready)$.pred,
    y_pred_rf = predict(rf_fit, new_data = play_test_ready)$.pred
  )

# ====================================================
# 9. Evaluate RMSE (if you have true output)
# ====================================================
rmse_x_lm <- rmse(play_test_preds$x, play_test_preds$x_pred_lm)
rmse_y_lm <- rmse(play_test_preds$y, play_test_preds$y_pred_lm)
overall_rmse_lm <- sqrt(mean((play_test_preds$x - play_test_preds$x_pred_lm)^2 +
                               (play_test_preds$y - play_test_preds$y_pred_lm)^2))

rmse_x_rf <- rmse(play_test_preds$x, play_test_preds$x_pred_rf)
rmse_y_rf <- rmse(play_test_preds$y, play_test_preds$y_pred_rf)
overall_rmse_rf <- sqrt(mean((play_test_preds$x - play_test_preds$x_pred_rf)^2 +
                               (play_test_preds$y - play_test_preds$y_pred_rf)^2))

cat("=== Linear Model ===\n")
cat("RMSE x:", rmse_x_lm, "\n")
cat("RMSE y:", rmse_y_lm, "\n")
cat("Overall RMSE:", overall_rmse_lm, "\n\n")

cat("=== Random Forest Model ===\n")
cat("RMSE x:", rmse_x_rf, "\n")
cat("RMSE y:", rmse_y_rf, "\n")
cat("Overall RMSE:", overall_rmse_rf, "\n")
