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
library(dplyr)

in_w09 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w09.csv")
out_w09 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w09.csv")

# ====================================================
# find which plays tyreek is in and the ones we want 
#to predict to make small subsets of data to work with 
# ====================================================
# tyreek <- in_w09 %>%
#   filter(player_name == "Tyreek Hill")
# 
# tyreek_play_counts <- tyreek %>%
#   filter(player_to_predict==TRUE) %>%
#   group_by(play_id) %>%
#   summarise(num_rows = n()) %>%
#   arrange(desc(num_rows))
#play_ id from greatest to least in # of frames 1102, 2645, 723, 3092, etc

game_id_to_use <- 2023110500
#game of miami vs chiefs (we think diff offenses)

# Uncomment whichever you want
play_input <- in_w09 %>%
#   filter(game_id == game_id_to_use)
  filter(play_id == 1102)

# singlegame_input <- vroom_write(play_input, "game_input.csv")

play_output <- out_w09 %>%
  # filter(game_id == game_id_to_use)
  filter(play_id == 1102)

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
  # ------------------------------------------------------------------
# 5-frame lag (0.5 s at 10 fps) – keep *all* players for now
# ------------------------------------------------------------------
mutate(
  across(
    c(x, y, vx, vy, ax, ay, s, a, o, dir,
      speed_sq, acc_sq, vx_vy_angle, dir_change,
      orientation_cos, orientation_sin, dir_cos, dir_sin),
    .fns = list(lag1 = ~lag(., 1),
                lag2 = ~lag(., 2),
                lag3 = ~lag(., 3),
                lag4 = ~lag(., 4),
                lag5 = ~lag(., 5)),
    .names = "{.col}_{.fn}"
  ),
  n_past = row_number() - 1
) %>%
  ungroup() %>%
  # later change this to use all players instead of only predictors
  filter(player_to_predict == TRUE)%>%
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
# add play_direction later
categorical_vars <- c("player_position", "player_role")

safe_factor <- function(col) {
  if (is.factor(col)) {
    lvls <- levels(col)
    keep <- tabulate(col, nbins = length(lvls)) > 1   # keep if >1 obs
    if (any(keep)) {
      factor(col, levels = lvls[keep])
    } else {
      # if *all* levels have ≤1 obs → return a constant dummy (0)
      factor(rep("CONST", length(col)), levels = "CONST")
    }
  } else {
    col
  }
}

# Apply the helper to each categorical column
cat_data <- play_features[categorical_vars] %>%
  mutate(across(everything(), safe_factor))

# # Dummy variables **per player** (keeps player-specific effects)
dummies <- model.matrix(~ . - 1, data = play_features[categorical_vars]) %>%
  as.data.frame()

lag_cols <- play_features %>%
  select(ends_with(c("_lag1", "_lag2", "_lag3", "_lag4", "_lag5"))) %>%
  names()

base_numeric <- play_features %>%
  select(vx, vy, ax, ay, s, a, o, dir,
         speed_sq, acc_sq, vx_vy_angle, dir_change,
         orientation_cos, orientation_sin, dir_cos, dir_sin,
         n_past,
         all_of(lag_cols))

predictors <- base_numeric %>% bind_cols(dummies)

fit_data <- bind_cols(
  predictors,
  play_features[, c("x_next", "y_next")]
) %>%
  drop_na()   # removes rows where any lag or lead is NA

# 3. Fit multivariate linear model -----------------------------------
# mv_lm <- lm(cbind(x_next, y_next) ~ ., data = bind_cols(predictors, play_features[, c("x_next", "y_next")]))
mv_lm <- lm(cbind(x_next, y_next) ~ ., data = fit_data)
     
 # TO DOUBLE CHECK IF MODEL IS PREDICTING WELL -----------------------
 plot(cbind(fit_data$x_next,fit_data$y_next), col = "black",xlim=c(45,90), ylim=c(0,35))
 # what the model is actually doing
 points(mv_lm$fitted.values, col='red')

# Join the SAME engineered columns (including 5-frame lags) to the output data
play_test <- play_output %>%
  left_join(
    play_features %>%
      select(game_id, play_id, nfl_id, frame_id,
             vx, vy, ax, ay, s, a, o, dir,
             speed_sq, acc_sq, vx_vy_angle, dir_change,
             orientation_cos, orientation_sin, dir_cos, dir_sin,
             n_past,
             all_of(lag_cols),
             all_of(categorical_vars)),
    by = c("game_id", "play_id", "nfl_id", "frame_id")
  )

# Apply the **same** safe-factor logic to the test set
cat_test <- play_test[categorical_vars] %>%
  mutate(across(everything(), safe_factor))

# Build dummy matrix for test set (same scheme as training)
dummies_test <- model.matrix(~ . - 1, data = play_test[categorical_vars]) %>% 
  as.data.frame()

test_predictors <- play_test %>%
  select(all_of(names(base_numeric))) %>%   # same columns as training
  bind_cols(dummies_test)

# 4. Generate features for output frames--------------------------------
# Join the same features to play_output
# play_test <- play_output %>%
#   left_join(
#     play_features %>%
#       select(game_id, play_id, nfl_id, frame_id, vx, vy, ax, ay, s, a, o, dir,
#              speed_sq, acc_sq, vx_vy_angle, dir_change,
#              orientation_cos, orientation_sin, dir_cos, dir_sin,
#              x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past,
#              play_direction, player_position, player_role),
#     by = c("game_id","play_id","nfl_id","frame_id")
#   )
# 
# # Build dummy matrix for test set
# dummies_test <- model.matrix(~ . - 1, data = play_test[categorical_vars]) %>% as.data.frame()
# test_predictors <- play_test %>%
#   select(vx, vy, ax, ay, s, a, o, dir,
#          speed_sq, acc_sq, vx_vy_angle, dir_change,
#          orientation_cos, orientation_sin, dir_cos, dir_sin,
#          x_lag1, y_lag1, vx_lag1, vy_lag1, ax_lag1, ay_lag1, n_past) %>%
#   bind_cols(dummies_test)

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


plot(x=play_output$x, y=play_output$y, col="black", xlim=c(45,90), ylim=c(0,35))
points(x=play_output_preds$x_pred, y=play_output_preds$y_pred, col="red")



# TRUE post-release positions (black)
plot(play_output_preds$x, play_output_preds$y,
     col = "black", pch = 16, cex = 0.8,
     xlim = c(0,120), ylim = c(0,53.5),
     xlab = "X (yards)", ylab = "Y (yards)",
     main = "True (black) vs Predicted (red) Post-Release Positions")

# PREDICTED post-release positions (red)
points(play_output_preds$x_pred, play_output_preds$y_pred,
       col = "red", pch = 16, cex = 0.8)

# Optional: connect true → predicted for each player
segments(play_output_preds$x, play_output_preds$y,
         play_output_preds$x_pred, play_output_preds$y_pred,
         col = "gray", lty = 2)

legend("topright", 
       legend = c("True Position", "Predicted Position"),
       col = c("black", "red"), pch = 16, cex = 0.9)


# Define field dimensions
field_length <- 120  # yards (includes 10-yard end zones)
field_width <- 53.3  # yards
# Yard numbers (excluding end zones)
yard_numbers <- c(10,20,30,40,50,40,30,20,10)
yard_positions <- seq(20, 100, by = 10)
# Base field plot
field_plot <- ggplot() +
  # Green field background
  geom_rect(aes(xmin = 0, xmax = field_length, ymin = 0, ymax = field_width),
            fill = "palegreen4", color = NA) +
  
  # End zones
  geom_rect(aes(xmin = 0, xmax = 10, ymin = 0, ymax = field_width),
            fill = "gray30", alpha = 0.6) +
  geom_rect(aes(xmin = 110, xmax = 120, ymin = 0, ymax = field_width),
            fill = "gray30", alpha = 0.6) +
  
  # Main yard lines (every 5 yards)
  geom_vline(xintercept = seq(10, 110, by = 5), color = "white", linewidth = 0.4, alpha = 0.6) +
  
  # Thick 10-yard lines
  geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linewidth = 0.9) +
  
  # Sidelines
  geom_hline(yintercept = c(0, field_width), color = "white", linewidth = 1.2) +
  
  # Yard numbers on each side
  annotate("text", x = yard_positions, y = 4, label = yard_numbers,
           color = "white", size = 5, angle = 0, fontface = "bold") +
  annotate("text", x = yard_positions, y = field_width - 4, label = yard_numbers,
           color = "white", size = 5, angle = 180, fontface = "bold") +
  
  # Hash marks every yard (between the numbers)
  geom_segment(aes(x = seq(11, 109, by = 1), xend = seq(11, 109, by = 1),
                   y = 0.4, yend = 1.0), color = "white", linewidth = 0.25) +
  geom_segment(aes(x = seq(11, 109, by = 1), xend = seq(11, 109, by = 1),
                   y = field_width - 0.4, yend = field_width - 1.0),
               color = "white", linewidth = 0.25) +
  
  # Coordinate settings
  scale_x_continuous(limits = c(0, field_length), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, field_width), expand = c(0, 0)) +
  coord_fixed() +
  
  theme_void() +
  ggtitle("NFL Field (Yards)")

play_input <- play_input %>%
  filter(player_to_predict == TRUE)

play_output <- play_output %>%
  left_join(play_input %>% select(nfl_id, player_name, player_side),
            by = "nfl_id")

field_plot +
  geom_point(data = play_input, aes(x = x, y = y, color = interaction("Input", player_side)), size = 1) +
  geom_point(data = play_input, aes(x = ball_land_x, y = ball_land_y), color = "yellow", size = 2) +
  geom_point(data = play_output, aes(x = x, y = y, color = interaction("Output", player_side)), size = 1) +
  scale_color_manual(
    values = c(
      "Input.Offense"  = "coral",
      "Input.Defense"  = "lightblue",
      "Output.Offense" = "red",
      "Output.Defense" = "blue"
    ),
    labels = c(
      "Input – Offense",
      "Input – Defense",
      "Output – Offense",
      "Output – Defense"
    ),
    name = "Phase / Team"
  )+ theme(legend.position = "bottom")

# # theme(legend.position = "none")
# play_output <- play_output %>%
#   left_join(play_input %>% select(nfl_id, player_name, player_side),
#             by = "nfl_id")