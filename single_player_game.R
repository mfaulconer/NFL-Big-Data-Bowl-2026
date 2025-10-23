# extracting week 11 and 1 player to predict

# library(tidyverse)
# library(tidymodels)
library(vroom)
library(dplyr)
library(DataExplorer)
# library(patchwork)
library(lubridate)  # for hour extraction
library(bonsai)
library(lightgbm)
library(slider)
library(data.table)
library(stringr)
library(lightgbm)

in_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\input_2023_w11.csv")
out_w11 <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\train\\output_2023_w11.csv")

players <- in_w11 %>%
  select(player_name, player_position, player_side, player_role) %>%
  distinct()
players <- in_w11 %>%
  select(player_name, player_position, player_side, player_role, player_to_predict) %>%
  distinct()
"C:\Users\Jenna\OneDrive\Documents\NFL-Big-Data-Bowl-2026"