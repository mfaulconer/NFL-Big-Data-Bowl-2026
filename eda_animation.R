# extracting week 11 and 1 player to predict

# library(tidyverse)
# library(tidymodels)
library(vroom)
library(dplyr)
# library(DataExplorer)
# library(patchwork)
# library(lubridate)  # for hour extraction
# library(bonsai)
# library(lightgbm)
# library(slider)
# library(data.table)
# library(stringr)
# library(lightgbm)
library(ggplot2)
library(gganimate)

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

# players <- in_w11 %>%
#   select(player_name, player_position, player_side, player_role) %>%
#   distinct()
players <- in_w11 %>%
  select(player_name, player_position, player_side, player_role, player_to_predict) %>%
  distinct()

#tyreek hill
#ajbrown

tyreek <- in_w11 %>%
  filter(player_name == "Tyreek Hill")

# play_id = 2860, 329, 1315, 1180

tyreek_input <- tyreek %>%
  filter(play_id == "1180")

# play_id 2860 only has 32 rows so i think 3.2 seconds (one of biggest info's we have)
tyreek_output <- out_w11 %>%
  filter(play_id == "1180")

tyreek_play_counts <- tyreek %>%
  group_by(play_id) %>%
  summarise(num_rows = n()) %>%
  arrange(desc(num_rows))

# ------------------------------------------------------------------------------------------
# plot field and tyreek's data so we can see 
# ------------------------------------------------------------------------------------------
play_input <- in_w11 %>%
  filter(play_id == 2860)  # replace with your actual play_id

# # Base field
# field_plot <- ggplot() +
#   # green field
#   geom_rect(aes(xmin = 0, xmax = 120, ymin = 0, ymax = 53.3),
#             fill = "palegreen4", color = "white") +
#   # end zones
#   geom_rect(aes(xmin = 0, xmax = 10, ymin = 0, ymax = 53.3),
#             fill = "gray30", alpha = 0.5) +
#   geom_rect(aes(xmin = 110, xmax = 120, ymin = 0, ymax = 53.3),
#             fill = "gray30", alpha = 0.5) +
#   # yard lines every 10 yards
#   geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linewidth = 0.5) +
#   # coordinate limits
#   scale_x_continuous(limits = c(0, 120)) +
#   scale_y_continuous(limits = c(0, 53.3)) +
#   coord_fixed() +
#   theme_void() +
#   ggtitle("NFL Field Coordinate System")

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

field_plot


field_plot +
  geom_point(data = play_input, aes(x = x, y = y, color = player_side), size = 1) +
  geom_point(data = play_input, aes(x=ball_land_x, y= ball_land_y), color = "yellow", size = 2)
  # theme(legend.position = "none")

## output of that plot above
play_output <- out_w11 %>%
  filter(play_id == 2860)  # replace with your actual play_id
play_output <- play_output %>%
  left_join(play_input %>% select(nfl_id, player_name, player_side),
            by = "nfl_id")

field_plot +
  geom_point(data = play_output, aes(x = x, y = y, color = player_side), size = 1) +
  geom_point(data = play_input, aes(x=ball_land_x, y= ball_land_y), color = "yellow", size = 2)
# theme(legend.position = "none")

# ------------------------------------------------------------------------------------------
# plot the entire play with tyreek hill 
# ------------------------------------------------------------------------------------------

play_input <- in_w11 %>%
  filter(play_id == 1180) %>%  # replace with your actual play_id
filter(player_name == "Tyreek Hill")

field_base <- ggplot() +
  # Field background
  geom_rect(aes(xmin = 0, xmax = 120, ymin = 0, ymax = 53.3),
            fill = "palegreen4", color = "white") +
  
  # End zones
  geom_rect(aes(xmin = 0, xmax = 10, ymin = 0, ymax = 53.3),
            fill = "gray30", alpha = 0.5) +
  geom_rect(aes(xmin = 110, xmax = 120, ymin = 0, ymax = 53.3),
            fill = "gray30", alpha = 0.5) +
  
  # Yard lines
  geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linewidth = 0.5) +
  scale_x_continuous(limits = c(0, 120)) +
  scale_y_continuous(limits = c(0, 53.3)) +
  coord_fixed() +
  theme_void()

# Animation
anim <- field_base +
  geom_point(data = play_input,
             aes(x = x, y = y, color = player_name),
             size = 3) +
  geom_point(data = play_input, aes(x = ball_land_x, y = ball_land_y), color = "yellow",
             size = 3) +
  labs(title = 'Play ID: {closest_state}') +
  transition_states(states = frame_id, transition_length = 2, state_length = 1) +
  ease_aes('linear')

# Render animation
animate(anim, nframes = max(play_input$frame_id)*2, renderer = gifski_renderer())

# ------------------------------------------------------------------------------------------
# plot after ball is released for specified play_id
# ------------------------------------------------------------------------------------------

play_output <- out_w11 %>%
  filter(play_id == 1180)  # replace with your actual play_id
play_output <- play_output %>%
  left_join(play_input %>% select(nfl_id, player_name),
            by = "nfl_id")

field_base <- ggplot() +
  # Field background
  geom_rect(aes(xmin = 0, xmax = 120, ymin = 0, ymax = 53.3),
            fill = "palegreen4", color = "white") +
  
  # End zones
  geom_rect(aes(xmin = 0, xmax = 10, ymin = 0, ymax = 53.3),
            fill = "gray30", alpha = 0.5) +
  geom_rect(aes(xmin = 110, xmax = 120, ymin = 0, ymax = 53.3),
            fill = "gray30", alpha = 0.5) +
  
  # Yard lines
  geom_vline(xintercept = seq(10, 110, by = 10), color = "white", linewidth = 0.5) +
  scale_x_continuous(limits = c(0, 120)) +
  scale_y_continuous(limits = c(0, 53.3)) +
  coord_fixed() +
  theme_void()

# Animation
anim <- field_base +
  geom_point(data = play_output,
             aes(x = x, y = y, color = player_name),
             size = 3) +
  geom_point(data = play_input, aes(x = ball_land_x, y = ball_land_y), color = "yellow",
             size = 3) +
  labs(title = 'Play ID: {closest_state}') +
  transition_states(states = frame_id, transition_length = 2, state_length = 1) +
  ease_aes('linear')

# Render animation
animate(anim, nframes = max(play_to_plot$frame_id)*2, renderer = gifski_renderer())


# ------------------------------------------------------------------------------------------
# plot after ball is released for specified play_id
# ------------------------------------------------------------------------------------------
