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

# Load data
# test_index <- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\test.csv")
test_input<- vroom("C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data\\test_input.csv")

# test_input will work just fine don't need test_index right now

# Define paths and constants
DATA_PATH <- "C:\\Users\\Jenna\\OneDrive\\Documents\\NFL-Big-Data-Bowl-2026\\data"
TRAIN_PATH <- file.path(DATA_PATH, "train")
WEEKS <- 1:18
# N_FOLDS <- 5
# RANDOM_STATE <- 42

# Function to load weekly data
load_weeks <- function(week_nums) {
  inputs <- list()
  outputs <- list()
  
  for (w in week_nums) {
    input_file <- file.path(TRAIN_PATH, sprintf("input_2023_w%02d.csv", w))
    output_file <- file.path(TRAIN_PATH, sprintf("output_2023_w%02d.csv", w))
    
    inputs[[length(inputs) + 1]] <- fread(input_file)
    outputs[[length(outputs) + 1]] <- fread(output_file)
  }
  
  input_df <- rbindlist(inputs, use.names = TRUE, fill = TRUE)
  output_df <- rbindlist(outputs, use.names = TRUE, fill = TRUE)
  
  return(list(input_df = input_df, output_df = output_df))
}

# Load data
data <- load_weeks(WEEKS)
input_df <- data$input_df
output_df <- data$output_df

# Summary
cat(sprintf("Data loaded: %d rows, %d games\n", nrow(input_df), uniqueN(input_df$game_id)))

DataExplorer::plot_correlation(input_df)
DataExplorer::plot_intro(input_df)
DataExplorer::plot_bar(input_df)
DataExplorer::plot_missing(output_df)
DataExplorer::plot_histogram(input_df)
