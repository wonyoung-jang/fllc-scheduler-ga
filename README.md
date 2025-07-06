# FLL-C Scheduler GA (Genetic Algorithm)

FLL-C Scheduler GA uses a multi-objective genetic algorithm (NSGA-II) to generate schedules for FIRST LEGO League Challenge.

## Features

-   **Multi-Objective Optimization:** Creates schedules that balance break time distribution, opponent variety, and table consistency.
-   **Configurable:** Tournament parameters (teams, rounds, locations, times) are defined in a `.ini` configuration file.
-   **Preflight Validation:** Checks configuration for probably-impossible-to-solve scenarios before starting.
-   **Export:** Outputs schedules in **CSV** and **HTML** formats.
-   **Visualization:** Automatically generates plots showing fitness evolution and the final Pareto front of optimal solutions.

## Getting Started

### Prerequisites

This project requires Python 3.13+ and the following packages:

-   `matplotlib`
-   `pandas`
-   `tqdm`

### Installation

1.  Clone repository:
    ```bash
    git clone https://github.com/wonyoung-jang/fllc-scheduler-ga.git
    cd fll-scheduler-ga
    ```

2.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Using the scheduler is a two-step process: creating a configuration file and running the main script.

### 1. Create a Configuration File

Create a file named `config.ini` to define your tournament's structure. The configuration is split into a `[DEFAULT]` section for global settings, a `[genetic]` section for the genetic algorithm's parameters, and one `[round.*]` section for each type of activity (e.g., judging, practice, official/table).

**Genetic Parameters:**

-   `population_size`: The population size (number of schedules) for each generation.
-   `generations`: The number of generations to run.
-   `elite_size`: The number of the best schedules to include in the next generation.
-   `selection_size`: The number of schedules that compete to be selected to evolve into the next generation.
-   `crossover_chance`: The chance of two parent schedules crossover breeding to create an offspring.
-   `mutation_chance_low`: The lower limit chance for an offspring schedule to mutate (adaptive). If they are better than the average of the population, they mutate based on this value.
-   `mutation_chance_high`: The higher limit chance for an offspring schedule to mutate (adaptive). If they are worse than the average of the population, they mutate based on this value.

**Round Parameters:**

-   `num_teams`: (Default) Total number of teams in the tournament.
-   `round_type`: Type of round. Must be `Judging`, `Practice`, or `Table`.
-   `rounds_per_team`: How many times each team must participate in this type of round.
-   `teams_per_round`: The number of teams involved in a single instance of this round (e.g., `1` for judging, `2` for a match).
-   `start_time`: The time of the very first slot for this round type (`HH:MM` 24-hour format).
-   `stop_time`: The time of the latest stop time for the last slot (`HH:MM` 24-hour format).
-   `duration_minutes`: The duration of a single round in minutes.
-   `num_locations`: Number of parallel locations for this round (e.g., 3 judging rooms or 4 competition tables).

**Example `config.ini`:**

```ini
[DEFAULT]
num_teams = 42

[genetic]
population_size = 16
generations = 128
elite_size = 2
selection_size = 4
crossover_chance = 0.5
mutation_chance_low = 0.2
mutation_chance_high = 0.8

[round.judging]
round_type = Judging
rounds_per_team = 1
teams_per_round = 1
start_time = 08:00
duration_minutes = 45
num_locations = 7 ; i.e., 7 judging rooms

[round.practice]
round_type = Practice
rounds_per_team = 2
teams_per_round = 2
start_time = 09:00
stop_time = 12:00 ; Add optional stop time to control buffer slots
duration_minutes = 15
num_locations = 4  ; Same tables as the official matches

[round.table]
round_type = Table
rounds_per_team = 3
teams_per_round = 2
start_time = 13:30
duration_minutes = 11
num_locations = 4  ; i.e., 4 competition tables (A, B, C, D)
```

### 2. Run the Scheduler

Execute the main script from your terminal, providing the path to your configuration and specifying an output file name. The file extension of the output (`.csv` or `.html`) will determine the export format.

```bash
python fll_scheduler_ga --config_file path/to/config.ini --output_dir schedule_results
```

The program will run the genetic algorithm, showing a progress bar for the generations.

The genetic parameters will come from the config.ini by default. We can overwrite the default parameters directly in the file or in a CLI:

```bash
python fll_scheduler_ga --population_size 500 --crossover_chance 0.9
```

### 3. Review the Output

After the run is complete, the following files will be created in the directory you specified:

-   `.csv` and `.html` files showing human readable schedules.
-   `fitness_plot.png`: A line graph showing how the average fitness of the best solutions improved over each generation.
-   `pareto_front.png`: A parallel coordinates plot showing the trade-offs between the different objectives for all optimal solutions found.
-   `pareto_front_scatter.png`: (For 2 or 3 objectives) A scatter plot visualizing the Pareto front.