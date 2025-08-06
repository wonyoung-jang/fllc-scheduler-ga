# FIRST LEGO League Challenge Scheduler NSGA-III (Non-dominated Sorting Genetic Algorithm III)

![Image of scatterplot](/assets/2022-2023/after_512_generations/pareto_scatter_3d.png)

FLL-C Scheduler NSGA-III uses a multi-objective optimization non-dominated sorting genetic algorithm (NSGA-III) to generate tournament schedules for FIRST LEGO League Challenge.

## [Click here for a run through of a comparison with a real FLLC schedule](/REAL_LIFE_EXAMPLES.md)

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
-   `numpy`
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

Create a file named `config.ini` to define your tournament's structure. The configuration is split into sections for different parts of the application.

**Team Parameters:**

-   `num_teams`: Total number of teams in the tournament.

**Location Parameters:**

-   `name`: The name of the location type.
-   `identities`: The identities for the locations.
-   `sides`: Determines whether or not a location is a match or solo area.
-   `teams_per_round`: How many teams are schedule for this location type per round.

**Round Parameters:**

-   `round_type`: Type of round. Must be `Judging`, `Practice`, or `Table`.
-   `rounds_per_team`: How many times each team must participate in this type of round.
-   `teams_per_round`: The number of teams involved in a single instance of this round (e.g., `1` for judging, `2` for a match).
-   `times`: A list of comma separated start times for the tournament.
-   `start_time`: The time of the very first slot for this round type (`HH:MM` 24-hour format).
-   `stop_time`: The time of the latest stop time for the last slot (`HH:MM` 24-hour format).
-   `duration_minutes`: The duration of a single round in minutes.
-   `location`: The location type determined by the `name` of the location associated with this round.

**Fitness Parameters:**

-   `weight_mean`: Mean weight - The average fitness of all teams.
-   `weight_variation`: Variation weight - How much the teams can vary in fitness.
-   `weight_range`: Range weight - How far apart the best and worst teams are.

**Genetic Parameters:**

-   `population_size`: The population size (number of schedules) for each generation.
-   `generations`: The number of generations to run.
-   `offspring_size`: The number of offspring schedules to produce each generation.
-   `selection_size`: The number of schedules that compete to be selected to evolve into the next generation.
-   `crossover_chance`: The chance of two parent schedules crossover breeding to create an offspring.
-   `mutation_chance`: The chance for an offspring schedule to mutate.
-   `seed`: Random seed for reliable output.
-   `genetic.operator.selection`: Configure GA selection.
-   `genetic.operator.crossover`: Configure GA crossover.
-   `genetic.operator.mutation`: Configure GA muatation.

**Example `config.ini`:**

```ini
[teams]
num_teams = 42

[fitness]
; Values of weights are porportional to each other, the following are the same: 
; (1, 2, 3) == (50, 100, 150) == (0.01, 0.02, 0.03)
; Each translates to: (1/6, 2/6, 3/6) ~~ (0.166, 0.333, 0.500)
weight_mean = 3         ; Mean weight - The average fitness of all teams
weight_variation = 1    ; Variation weight - How much the teams can vary in fitness
weight_range = 1        ; Range weight - How far apart the best and worst teams are

[location.room]
name = Room
identities = 1, 2, 3, 4, 5, 6, 7
sides = 1
teams_per_round = 1

[location.table]
name = Table
identities = A, B, C, D
sides = 2
teams_per_round = 2

[round.judging]
round_type = Judging
rounds_per_team = 1
teams_per_round = 1
start_time = 08:00
duration_minutes = 45
location = Room

[round.practice]
round_type = Practice
rounds_per_team = 2
teams_per_round = 2
start_time = 09:00
stop_time = 12:00 ; Add optional stop time to control buffer slots
duration_minutes = 15
location = Table

[round.table]
round_type = Table
rounds_per_team = 3
teams_per_round = 2
; start_time = 13:30
; Optional: times can be specified instead of start_time as a comma-separated list
times = 
  13:30,
  13:41,
  13:52,
  14:03,
  14:14,
  14:25,
  14:36,
  14:47,
  14:58,
  15:09,
  15:20,
  15:31,
  15:42,
  15:53,
  16:04,
  16:15
duration_minutes = 11
location = Table

[genetic]
population_size = 16
generations = 512
offspring_size = 12
selection_size = 6
crossover_chance = 0.5
mutation_chance = 0.5
seed = ; Optional seed for random number generation, leave empty for random seed, or input an integer value
num_islands = 10
migration_interval = 10
migration_size = 4

[genetic.operator.selection]
; Available selection types: TournamentSelect, RandomSelect
selection_types = TournamentSelect, RandomSelect

[genetic.operator.crossover]
; Available crossover types: 
;   - KPoint                (uses crossover_ks to determine split points)
;   - Scattered             (swaps random half of events between parents)
;   - Uniform               (swaps each gene randomly)
;   - RoundTypeCrossover    (preserves whole round types during crossover)
;   - PartialCrossover      (partially take genes from both parents, repair to complete the child)
crossover_types = KPoint, Scattered, Uniform, RoundTypeCrossover, PartialCrossover
; Split point values for KPoint crossover, does nothing if KPoint is not used
crossover_ks = 1, 2, 4 

[genetic.operator.mutation]
; Available mutation types:
;   - SwapMatch_CrossTimeLocation (swaps entire matches across different times and locations)
;   - SwapMatch_SameLocation      (swaps matches in the same location but at different times)
;   - SwapMatch_SameTime          (swaps matches at the same time but in different locations)
;   - SwapTeam_CrossTimeLocation  (swaps single teams across different times and locations)
;   - SwapTeam_SameLocation       (swaps single teams in the same location but at different times)
;   - SwapTeam_SameTime           (swaps single teams at the same time but in different locations)
; Multiline string for better readability - MUST be indented
mutation_types = 
    SwapMatch_CrossTimeLocation, 
    SwapMatch_SameLocation, 
    SwapMatch_SameTime, 
    SwapTeam_CrossTimeLocation, 
    SwapTeam_SameLocation, 
    SwapTeam_SameTime
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
-   `fitness_vs_generation.png`: A line graph showing how the average fitness of the best solutions improved over each generation.
-   `pareto_front.png`: A parallel coordinates plot showing the trade-offs between the different objectives for all optimal solutions found.
-   `pareto_scatter_(2d or 3d).png`: (For 2 or 3 objectives) A scatter plot visualizing the Pareto front.