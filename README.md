# FIRST LEGO League Challenge Scheduler NSGA-III (Non-dominated Sorting Genetic Algorithm III)

![Image of scatterplot](/assets/2022-2023/after_512_generations/pareto_scatter_3d.png)

FLL-C Scheduler NSGA-III uses a multi-objective optimization non-dominated sorting genetic algorithm (NSGA-III) to generate tournament schedules for FIRST LEGO League Challenge.

## [Click here for a run through of a comparison with a real FLLC schedule](/REAL_LIFE_EXAMPLES.md)

## Features

-   **Multi-Objective Optimization:** Creates schedules that balance break time distribution, opponent variety, and table consistency.
-   **Configurable:** Tournament parameters (teams, rounds, locations, times) are defined in a `json` configuration file.
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

Create a file named `config.json` to define your tournament's structure. The configuration is split into sections for different parts of the application.

#### Genetic Parameters:

-   `generations`: Number of generations to run.
-   `population_size`: Population size (number of schedules) for each generation.
-   `offspring_size`: Number of offspring schedules to produce each generation.
-   `crossover_chance`: Chance of two parent schedules to crossover to create an offspring.
-   `mutation_chance`: Chance for an offspring schedule to mutate.
-   `num_islands`: Number of parallel sub-populations (islands) evolved independently.
-   `migration_interval`: Number of generations between migrations of individuals between islands.
-   `migration_size`: Number of individuals exchanged during each migration event.
-   `rng_seed`: Random seed for reliable output.

#### Genetic Operators:

##### Crossover

-   `types`: List of crossover operator names to sample from when breeding (e.g., `KPoint`, `Scattered`, `Uniform`, `RoundTypeCrossover`, `TimeSlotCrossover`, `LocationCrossover`).
-   `k_vals`: Allowed values for K when using `KPoint` crossover (e.g., `[1, 2, 3]`).

##### Mutation

-   `types`: List of mutation operator names to sample from (e.g., `SwapMatch_CrossTimeLocation`, `SwapMatch_SameLocation`, `SwapMatch_SameTime`, `SwapTeam_CrossTimeLocation`, `SwapTeam_SameLocation`, `SwapTeam_SameTime`, `SwapTableSide`, `Inversion`, `Scramble`).

#### Runtime:

-   `add_import_to_population`: If true, add the imported schedule (from `import_file`) into the initial population.
-   `flush`: If true, clear previous run outputs in the `output_dir` before starting.
-   `flush_benchmarks`: If true, clear previously saved fitness benchmark objects before starting.
-   `import_file`: Path to an existing schedule CSV to seed/compare or include in the initial population.
-   `seed_file`: Path to a saved run/checkpoint (pickle) used to resume or bootstrap the population.

#### Imports:

-   `seed_pop_sort`: If set to **"random"**, seed schedules will be added to islands in random order. If set to **"best"**, seed schedules will be added to islands in order of the sum of their fitness values. 
-   `seed_island_strategy`: If set to **"distributed"**, each island is added to at similar rates. If set to **"concentrated"**, an island is filled before filling the next one.

#### Exports:

-   `output_dir`: Directory where all outputs (exports, plots, logs) are written.
-   `summary_reports`: If true, write human-readable run summaries/statistics.
-   `schedules_csv`: If true, export schedules as CSV.
-   `schedules_html`: If true, export schedules as HTML.
-   `schedules_team_csv`: If true, export per-team schedule CSVs.
-   `pareto_summary`: If true, export a compact summary (e.g., CSV) of the Pareto front metrics.
-   `plot_fitness`: If true, save fitness vs. generation plots.
-   `plot_parallel`: If true, save a parallel-coordinates plot of objective trade-offs.
-   `plot_scatter`: If true, save a 2D/3D scatter plot of the Pareto front (for 2–3 objectives).
-   `front_only`: If true, export schedules and summaries of only the non-dominated (Pareto-front) solutions; otherwise include the whole final population.
-   `no_plotting`: If true, disable all plotting.
-   `cmap_name`: Matplotlib colormap name to use for plots (e.g., `viridis`, `plasma`).

#### Logging:

-   `log_file`: Name/path of the log file.
-   `loglevel_file`: Logging level for file output. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. 
-   `loglevel_console`: Logging level for console output. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.

#### Team:

-   `teams`: Total number of teams, or a comma separated list of team names or identities.

#### Fitness:

-   `weight_mean`: Average fitness of all teams.
-   `weight_variation`: How much team fitnesses can vary.
-   `weight_range`: How far apart the best and worst team fitnesses are.

#### Time:

-   `format`: 12 or 24 hour format.

#### Location:

-   `name`: Name of the location type.
-   `count`: Number of locations of this type.
-   `sides`: Determines whether or not a location is a match or solo area and how many teams per round.

#### Round:

-   `roundtype`: Type of round. Must be `Judging`, `Practice`, or `Table`.
-   `location`: Location type determined by the `name` of the location associated with this round.
-   `rounds_per_team`: How many times each team must participate in this type of round.
-   `teams_per_round`: Number of teams involved in a single instance of this round (e.g., `1` for judging, `2` for a match).
-   `times`: List of comma separated start times for the tournament.
-   `start_time`: Time of the very first slot for this round type.
-   `stop_time`: Time of the latest stop time for the last slot.
-   `duration_minutes`: Duration of a single round in minutes.

**Example `config.json`:**

```json
{
  "genetic": {
    "parameters": {
      "generations": 5000,
      "population_size": 32,
      "offspring_size": 4,
      "crossover_chance": 0.7,
      "mutation_chance": 0.4,
      "num_islands": 8,
      "migration_interval": 100,
      "migration_size": 3,
      "rng_seed": null
    },
    "operator": {
      "crossover": {
        "types": ["KPoint", "Scattered", "Uniform", "RoundTypeCrossover", "TimeSlotCrossover", "LocationCrossover"],
        "k_vals": [1, 2, 3]
      },
      "mutation": {
        "types": [
          "SwapMatch_CrossTimeLocation",
          "SwapMatch_SameLocation",
          "SwapMatch_SameTime",
          "SwapTeam_CrossTimeLocation",
          "SwapTeam_SameLocation",
          "SwapTeam_SameTime",
          "SwapTableSide",
          "Inversion",
          "Scramble"
        ]
      }
    }
  }
  "runtime": {
    "add_import_to_population": true,
    "flush": false,
    "flush_benchmarks": false,
    "import_file": "",
    "seed_file": "fll_scheduler_ga.pkl"
  },
  "imports": {
    "seed_pop_sort": "random",
    "seed_island_strategy": "distributed"
  },
  "exports": {
    "output_dir": "fllc_schedule_outputs",
    "summary_reports": true,
    "schedules_csv": true,
    "schedules_html": true,
    "schedules_team_csv": true,
    "pareto_summary": true,
    "plot_fitness": true,
    "plot_parallel": true,
    "plot_scatter": true,
    "front_only": true,
    "no_plotting": false,
    "cmap_name": "viridis"
  },
  "logging": {
    "log_file": "fll_scheduler_ga.log",
    "loglevel_file": "DEBUG",
    "loglevel_console": "INFO"
  },
  "teams": {
    "teams": 42
  },
  "fitness": {
    "weight_mean": 100,
    "weight_variation": 10,
    "weight_range": 0
  },
  "time": {
    "format": 24
  },
  "locations": [
    {
      "name": "Room",
      "count": 7,
      "sides": 1
    },
    {
      "name": "Table",
      "count": 4,
      "sides": 2
    }
  ],
  "rounds": [
    {
      "roundtype": "Judging",
      "location": "Room",
      "rounds_per_team": 1,
      "teams_per_round": 1,
      "start_time": "08:00",
      "stop_time": "12:30"
    },
    {
      "roundtype": "Practice",
      "location": "Table",
      "rounds_per_team": 2,
      "teams_per_round": 2,
      "times": ["09:00", "09:15", "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45"],
      "duration_minutes": 15
    },
    {
      "roundtype": "Table",
      "location": "Table",
      "rounds_per_team": 3,
      "teams_per_round": 2,
      "start_time": "13:30",
      "duration_minutes": 11
    }
  ]
}
```

### 2. Run the Scheduler

Execute the main script from your terminal, providing the path to your configuration and specifying an output file name. The file extension of the output (`.csv` or `.html`) will determine the export format.

```bash
uv run fll_scheduler_ga
```

The program will run the genetic algorithm, showing a progress bar for the generations.

### 3. Review the Output

After the run is complete, the following files will be created in the directory you specified:

-   `.csv` and `.html` files showing human readable schedules.
-   `fitness_vs_generation.png`: A line graph showing how the average fitness of the best solutions improved over each generation.
-   `pareto_front.png`: A parallel coordinates plot showing the trade-offs between the different objectives for all optimal solutions found.
-   `pareto_scatter_(2d or 3d).png`: (For 2 or 3 objectives) A scatter plot visualizing the Pareto front.