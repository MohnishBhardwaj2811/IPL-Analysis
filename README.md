
# IPL DATA ANALYSIS




## Installing Modules

Install my-project with npm

```bash
    pip install pandas
    pip install numpy
    pip install seaborn
    pip install matplotlib
    pip install ipywidgets
    pip install ipython
    pip install Pillow
    pip install requests
```

## What Each Module Does:

 ```bash 
 pip install pandas
 ```

Used for data manipulation and analysis. It allows loading CSV files, filtering data, grouping, and performing complex data operations with ease using DataFrames.

 ```bash 
 pip install numpy
 ``` 

Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. It's often used for numerical calculations and statistical operations.

 ```bash 
 pip install seaborn
 ``` 

A high-level data visualization library based on matplotlib. It offers an easier way to create beautiful and informative statistical plots.

 ```bash 
 pip install matplotlib
```

The foundational plotting library in Python. It is used to create static, animated, and interactive plots and gives complete control over graph styling and customization.

```bash
pip install ipywidgets
```

Enables interactive widgets like sliders, dropdowns, and checkboxes inside Jupyter notebooks or Google Colab. Useful for building dynamic dashboards.

ipython: 

Enhances the interactive computing experience. It powers Jupyter notebooks and provides features like pretty printing, magics (e.g., %timeit), and better error handling.


```bash
pip install  requests
```

A simple HTTP library used for sending requests to web servers. It helps in fetching data from APIs or downloading files from the internet.

```bash
from cycler import cycler

# IPL-dashboard palette mapping
ipl_palette = {
    # UI layers
    "background_main":  {"hex": "#0A2642"},
    "tile_background":  {"hex": "#092E50"},

    # Text / iconography
    "text_primary":  {"hex": "#FFFFFF"},
    "text_accent":   {"hex": "#ffffff"},   # #ffffff  #2F88B7

    # Data series
    "bar_primary":   {"hex": "#EAA43C"},
    "bar_secondary": {"hex": "#D97030"},
    "bar_tertiary":  {"hex": "#2F88B7"},

    # Outline / grid
    "outline":       {"hex": "#0A2642"},
}

# ─────────────────────────────────────────────────────────────
#  Custom theme setter
# ─────────────────────────────────────────────────────────────
def set_ipl_theme(pal: dict = ipl_palette) -> None:
    """
    Apply a dark-navy theme that matches the IPL dashboard artwork.
    Works for both Matplotlib and Seaborn.
    """
    # Convenience vars
    bg_main  = pal["background_main"]["hex"]
    bg_tile  = pal["tile_background"]["hex"]
    txt_prim = pal["text_primary"]["hex"]
    txt_acc  = pal["text_accent"]["hex"]
    outline  = pal["outline"]["hex"]

    # Colour cycle for data series
    data_cycle = [
        pal["bar_primary"]["hex"],
        pal["bar_secondary"]["hex"],
        pal["bar_tertiary"]["hex"]
    ]

    # ── Matplotlib rcParams ────────────────────────────────
    rc = {
        # Figure / axes face-colours
        "figure.facecolor":     bg_main,
        "axes.facecolor":       bg_tile,

        # Grid & spines
        "grid.color":           outline,
        "axes.edgecolor":       outline,

        # Text / ticks
        "text.color":           txt_prim,
        "axes.labelcolor":      txt_acc,
        "xtick.color":          txt_acc,
        "ytick.color":          txt_acc,

        # Legend styling
        "legend.facecolor": "none",           # Transparent background
        "legend.edgecolor": pal["outline"]["hex"],
        "legend.labelcolor": "#FFFFFF",       # ✅ Legend text color white
        "legend.title_fontsize": 11,
        "legend.fontsize": 10,


        # ─── NEW: single-plot padding ───
        "axes.xmargin": 0.05,   # 5 % blank on left/right
        "axes.ymargin": 0.10,   # 10 % blank on top/bottom

        # ─── NEW: default subplot padding ───
        "figure.subplot.left":   0.08,
        "figure.subplot.right":  0.97,
        "figure.subplot.top":    0.95,
        "figure.subplot.bottom": 0.07,
        "figure.subplot.wspace": 0.30,
        "figure.subplot.hspace": 0.30,

        # Font sizes
        "axes.titlesize":       14,
        "axes.labelsize":       12,
        "font.size":            11,

        # Colour cycle
        "axes.prop_cycle": cycler(color=data_cycle)
    }
    plt.rcParams.update(rc)

    # ── Seaborn global theme (inherits Matplotlib rc) ──────
    sns.set_theme(
        style="dark",           # minimal gridlines
        rc={
            "figure.facecolor": bg_main,
            "axes.facecolor":   bg_tile,
            "grid.color":       outline,
            "axes.labelcolor":  txt_acc,
            "xtick.color":      txt_acc,
            "ytick.color":      txt_acc,
        }
    )



if __name__ == "__main__":
    set_ipl_theme()
```

## 📂 Theme Module – Point‑wise Explanation

### 1. Imports
- `from cycler import cycler`  
  *Matplotlib helper that lets you define custom colour cycles (list of colours to loop through when plotting multiple series).*

### 2. `ipl_palette` (dictionary)
A single source of truth for every colour used in the dashboard theme.
- **UI layers**  
  - `background_main` – deep navy (#0A2642) for full‑figure background.  
  - `tile_background` – slightly lighter navy (#092E50) for axes/panel backgrounds.
- **Text / iconography**  
  - `text_primary` – pure white for high‑contrast titles.  
  - `text_accent` – white (placeholder for an accent colour if needed).
- **Data series (bars/lines)**  
  - `bar_primary` – amber (#EAA43C).  
  - `bar_secondary` – orange (#D97030).  
  - `bar_tertiary` – teal (#2F88B7).
- **Outline / grid**  
  - `outline` – same dark navy as main background, keeps grid subtle.

### 3. `set_ipl_theme(pal: dict = ipl_palette) -> None`
Configures Matplotlib **and** Seaborn in one call.

1. **Extract convenience variables** – grabs hex codes from the palette so you don’t hard‑code colours again.
2. **Build `data_cycle`** – list of the three bar/line colours injected into `axes.prop_cycle`.
3. **Prepare `rc` dict** – all Matplotlib style overrides in one place:
   - *Figure & Axes colours* – sets dark navy and tile navy backgrounds.
   - *Grid & Spines* – colour set to `outline`.
   - *Text & Ticks* – white for body text, accent colour for labels/ticks.
   - *Legend* – transparent background, white text, navy border.
   - *Margins* – 5 % X margin, 10 % Y margin so bars never touch edges.
   - *Default subplot paddings* – neat whitespace for multi‑axes layouts.
   - *Font sizes* – titles 14 pt, labels 12 pt, base font 11 pt.
   - *Colour cycle* – plugs `data_cycle` into Matplotlib via `cycler`.
4. **Update global `plt.rcParams`.**
5. **Tell Seaborn** to adopt the same face colours, label colours, and grids (`sns.set_theme(..., style="dark")`).

> *Call this **once** per notebook/kernel—every subsequent `plt.figure()` inherits the IPL look.*

### 4. `if __name__ == "__main__": …`
- Allows the file to be run standalone (`python theme.py`) for a quick visual test—it simply calls `set_ipl_theme()`.

### 5. External Requirement
- `cycler` – install with `pip install cycler` (usually bundled with Matplotlib).


## Functions for plotting Player stats 

```bash
# Variable Function


def all_players_as_list(deliveries_df):
    """
    Return a **list** containing every distinct player name that appears
    in the IPL deliveries data—taken from the columns
    'batter', 'bowler', and 'non_striker'.

    • Any missing (NaN) entries are ignored.
    • The function returns the names **sorted alphabetically**
      (remove the sorted() wrapper if you prefer the raw order).
    """
    cols = ["batter", "bowler", "non_striker"]

    # Use a Python set for uniqueness, then convert to list
    unique_names = (
        pd.concat([deliveries_df[c] for c in cols])   # stack the 3 columns
          .dropna()                                   # ditch NaNs
          .unique()                                   # NumPy array of uniques
    )

    return sorted(unique_names)     # turn into an alphabetically‑sorted list

def extract_team_names(deliv_df, matches_df=None):
    """
    Return an alphabetically‑sorted list of all unique IPL team names found in:

      • deliveries columns:  'batting_team', 'bowling_team'
      • matches columns (optional):  'team1', 'team2'

    Parameters
    ----------
    deliv_df   : pandas.DataFrame
        The deliveries data frame (mandatory).
    matches_df : pandas.DataFrame or None
        The matches data frame (optional – pass it if you have it).

    Example
    -------
    teams = extract_team_names(deliveries, matches)
    """
    team_set = set()

    # From deliveries.csv
    for col in ["batting_team", "bowling_team"]:
        if col in deliv_df.columns:
            team_set.update(deliv_df[col].dropna().unique())

    # From matches.csv (if supplied)
    if matches_df is not None:
        for col in ["team1", "team2"]:
            if col in matches_df.columns:
                team_set.update(matches_df[col].dropna().unique())

    return sorted(team_set)

def extract_all_umpires(matches_df: pd.DataFrame) -> list[str]:
    """
    Return a sorted list of every distinct umpire who appears
    in any of the umpire columns of the matches DataFrame.

    Expected umpire columns can be 'umpire1', 'umpire2', 'umpire3'
    (or similar).  The function auto‑detects any column whose name
    begins with 'umpire' (case‑insensitive).
    """
    # Identify all columns that start with "umpire"
    umpire_cols = [c for c in matches_df.columns
                   if c.lower().startswith("umpire")]

    # Pull values → flatten → drop NaN → unique → sort
    all_umpires = (
        pd.Series(matches_df[umpire_cols].values.ravel())
          .dropna()
          .unique()
          .tolist()
    )
    return sorted(all_umpires)


# ────────────────────────────────────────────────────────────
# 2️⃣  Get every season present in deliveries.csv
# ────────────────────────────────────────────────────────────
def extract_all_seasons(deliveries_df: pd.DataFrame) -> list[int]:
    """
    Return a sorted list of all distinct seasons in the deliveries
    DataFrame. Assumes a 'season' column exists; if not, you can
    derive it by merging with the matches table on match_id.
    """
    if "season" not in deliveries_df.columns:
        raise KeyError(
            "'season' column not found in deliveries_df. "
            "Merge deliveries with matches to add it."
        )

    seasons = (
        deliveries_df["season"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(seasons)


# variables

url_deleveries = "https://raw.githubusercontent.com/MohnishBhardwaj2811/IPL-Analysis/refs/heads/main/deliveries.csv"
url_matches    = "https://raw.githubusercontent.com/MohnishBhardwaj2811/IPL-Analysis/refs/heads/main/matches.csv"

#Loading And Cleaning Of Data
delivieres_dataset = pd.read_csv(url_deleveries)
matches_dataset    = pd.read_csv(url_matches)


# Intiating Some Important Variable

list_of_player = all_players_as_list(delivieres_dataset)
list_of_team   = extract_team_names(delivieres_dataset,matches_dataset)

list_of_umpire = extract_all_umpires(matches_dataset)
list_of_season = extract_all_seasons(matches_dataset)

list_of_metric = []


# Mapping and Cleaning of Data
delivieres_dataset["extras_type"] = delivieres_dataset["extras_type"].fillna("")
season_map = matches_dataset.set_index("id")["season"]
delivieres_dataset["season"] = delivieres_dataset["match_id"].map(season_map)


```

## 🧠 Variable and Utility Functions

This section documents helper functions and core variables used for player/team/umpire extraction, season detection, and dataset preparation.

---

### 🔹 Function: `all_players_as_list(deliveries_df)`
Returns an **alphabetically sorted list** of all unique player names found in the `batter`, `bowler`, and `non_striker` columns of the deliveries dataset.

- Ignores `NaN` values.
- Ensures no duplicates using `.unique()`.
- Sorts the result alphabetically for easier dropdown/menu building.

---

### 🔹 Function: `extract_team_names(deliv_df, matches_df=None)`
Returns a **combined list of all IPL team names** found in both:
- `deliveries.csv`: from `batting_team` and `bowling_team`.
- `matches.csv`: from `team1` and `team2` (optional).

Ensures uniqueness using a `set`, handles `NaN`, and returns a sorted list.

---

### 🔹 Function: `extract_all_umpires(matches_df)`
Detects and returns a **sorted list of unique umpire names** from all columns that start with `"umpire"` (case-insensitive).

- Flattens the relevant columns.
- Removes `NaN`.
- Returns a sorted list of all umpire names.

---

### 🔹 Function: `extract_all_seasons(deliveries_df)`
Returns a **sorted list of unique IPL seasons** present in the `deliveries_df`.

- Assumes there is a `season` column in the dataframe.
- If missing, suggests merging with `matches_df` using `match_id`.

> If the column `season` doesn't exist, it raises a `KeyError`.

---

## 📦 Core Variables and Dataset Loading

### 🔸 Data Source URLs
```python
url_deleveries = "https://raw.githubusercontent.com/MohnishBhardwaj2811/IPL-Analysis/refs/heads/main/deliveries.csv"
url_matches    = "https://raw.githubusercontent.com/MohnishBhardwaj2811/IPL-Analysis/refs/heads/main/matches.csv"
```

## 📥 Dataset Loading and Variable Initialization

---

## 🔸 Dataset Loading

```python
delivieres_dataset = pd.read_csv(url_deleveries)
matches_dataset    = pd.read_csv(url_matches)
```
Loads the CSV files directly from GitHub into pandas DataFrames:

delivieres_dataset contains ball-by-ball data.

matches_dataset contains match-level information.

## 🔸 Variables Created from the Datasets
```bash
list_of_player = all_players_as_list(delivieres_dataset)
```
All unique player names found across the columns: batter, bowler, and non_striker.

```bash
list_of_team = extract_team_names(delivieres_dataset, matches_dataset)
```
All unique team names found in the deliveries and matches datasets.

```bash
list_of_umpire = extract_all_umpires(matches_dataset)
```
All distinct umpire names found across columns starting with 'umpire'.

```bash
list_of_season = extract_all_seasons(matches_dataset)
```
Unique list of IPL seasons (e.g., 2008–2023) present in the dataset.


## 🔸 Data Cleaning and Mapping
```bash
# Filling missing extras with empty string
delivieres_dataset["extras_type"] = delivieres_dataset["extras_type"].fillna("")

# Mapping season values from match_id using matches dataset
season_map = matches_dataset.set_index("id")["season"]
delivieres_dataset["season"] = delivieres_dataset["match_id"].map(season_map)
```

Missing entries in the extras_type column are filled with empty strings for consistency.

The season column is created in the deliveries dataset by mapping match_id to the corresponding season from the matches dataset.

## Player plots

```python
# Player Ploting

# Function


def batsman_statistics(dataframe: pd.DataFrame, batter_name: str, consistency_threshold: int = 30):
    bf = dataframe[dataframe["batter"] == batter_name].copy()

    # Core aggregates
    runs = int(bf["batsman_runs"].sum())

    legal_mask = ~bf["extras_type"].isin(["wides"])
    balls_faced = int(legal_mask.sum())

    strike_rate = (runs / balls_faced * 100) if balls_faced else np.nan

    # Boundaries
    fours = int((bf["batsman_runs"] == 4).sum())
    sixes = int((bf["batsman_runs"] == 6).sum())
    boundary_runs = fours * 4 + sixes * 6

    # Dot balls (legal + 0 runs off the bat)
    dot_balls = int((legal_mask & (bf["batsman_runs"] == 0)).sum())
    dot_ball_pct = (dot_balls / balls_faced * 100) if balls_faced else np.nan

    # Per‑match aggregates
    match_runs = bf.groupby("match_id")["batsman_runs"].sum()
    matches_played = len(match_runs)

    fifties = int(match_runs[(match_runs >= 50) & (match_runs < 100)].count())
    hundreds = int(match_runs[match_runs >= 100].count())
    fifties_hundreds = f"{fifties} / {hundreds}"

    high_score = int(match_runs.max()) if matches_played else np.nan
    runs_per_match = (runs / matches_played) if matches_played else np.nan

    # Consistency: % of matches with runs >= threshold
    consistency = (
        match_runs[match_runs >= consistency_threshold].count() / matches_played * 100
        if matches_played else np.nan
    )

    # Dismissal information
    outs_mask = (dataframe["is_wicket"] == 1) & (dataframe["player_dismissed"] == batter_name)
    outs = int(outs_mask.sum())
    batting_average = (runs / outs) if outs else np.nan

    dismissal_types = (
        dataframe[outs_mask]["dismissal_kind"].value_counts().to_dict()
    )

    # Derived metrics
    boundary_pct = (boundary_runs / runs * 100) if runs else np.nan

    # Simple custom impact metric: (Runs per Match) * Strike Rate
    batting_impact_score = runs_per_match * strike_rate if not np.isnan(runs_per_match) else np.nan

    return {
        "Runs Scored": runs,
        "Balls Faced": balls_faced,
        "Strike Rate": round(strike_rate, 2),
        "Batting Average": round(batting_average, 2) if not np.isnan(batting_average) else np.nan,
        "Fours": fours,
        "Sixes": sixes,
        "Dot Ball %": round(dot_ball_pct, 2),
        "50s / 100s": fifties_hundreds,
        "Dismissal Types": dismissal_types,
        "Consistency": round(consistency, 2),
        "Boundary %": round(boundary_pct, 2),
        "Runs per Match": round(runs_per_match, 2),
        "High Score": high_score,
        "Batting Impact Score": round(batting_impact_score, 2),
    }


def bowler_statistics(df: pd.DataFrame, bowler_name: str):
    """
    Return a dict of advanced bowling metrics for `bowler_name`.
    """
    bdf = df[df["bowler"] == bowler_name].copy()
    bdf["extras_type"] = bdf["extras_type"].fillna("")

    # Legal deliveries
    legal_mask  = ~bdf["extras_type"].isin(["wides", "noballs"])
    legal_balls = int(legal_mask.sum())
    overs_float = legal_balls / 6

    # Wickets credited to bowler
    credited_kinds = {
        "bowled", "caught", "caught and bowled", "lbw",
        "stumped", "hit wicket", "hit-wicket", "caught & bowled"
    }
    wkt_mask = (bdf["is_wicket"] == 1) & (bdf["dismissal_kind"].isin(credited_kinds))
    wickets  = int(wkt_mask.sum())

    runs_conceded = int(bdf["total_runs"].sum())
    economy       = runs_conceded / overs_float if overs_float else np.nan
    bowl_avg      = runs_conceded / wickets if wickets else np.nan
    strike_rate   = legal_balls / wickets if wickets else np.nan

    # Dot‑ball %
    dots    = int((legal_mask & (bdf["total_runs"] == 0)).sum())
    dot_pct = dots / legal_balls * 100 if legal_balls else np.nan

    # Maiden overs
    over_totals  = bdf.groupby(["match_id", "inning", "over"])["total_runs"].sum()
    maiden_overs = int((over_totals == 0).sum())

    # Best bowling in a single match-inning
    inn_wkts = bdf.groupby(["match_id", "inning"])["dismissal_kind"].apply(
        lambda x: x.isin(credited_kinds).sum())
    inn_runs = bdf.groupby(["match_id", "inning"])["total_runs"].sum()

    df_best = pd.concat([inn_wkts, inn_runs], axis=1, keys=["wkts", "runs"])

    if not df_best.empty:
        best_row = df_best.sort_values(["wkts", "runs"], ascending=[False, True]).iloc[0]
        best_figs = f"{int(best_row.wkts)}/{int(best_row.runs)}"
    else:
        best_figs = "0/0"

    # Hauls
    four_fers = int((inn_wkts == 4).sum())
    five_fers = int((inn_wkts >= 5).sum())

    # Extras
    wides    = int((bdf["extras_type"] == "wides").sum())
    no_balls = int((bdf["extras_type"] == "noballs").sum())

    # Custom impact score
    impact = (wickets * 24) / (economy + 1) if not np.isnan(economy) and economy != -1 else np.nan

    return {
        "Wickets Taken": wickets,
        "Overs Bowled": round(overs_float, 2),
        "Economy Rate": round(economy, 2),
        "Bowling Average": round(bowl_avg, 2) if wickets else np.nan,
        "Bowling Strike Rate": round(strike_rate, 2) if wickets else np.nan,
        "Dot Ball %": round(dot_pct, 2),
        "Maiden Overs": maiden_overs,
        "Best Bowling Figures": best_figs,
        "4-Wicket Hauls": four_fers,
        "5-Wicket Hauls": five_fers,
        "Wide Balls": wides,
        "No Balls": no_balls,
        "Runs Conceded": runs_conceded,
        "Bowling Impact Score": round(impact, 2) if not np.isnan(impact) else np.nan,
    }



def feilder_statistics(df: pd.DataFrame, fielder_name: str):
    """
    Return a dict of fielding metrics for `fielder_name`.
    """
    df["fielder"] = df["fielder"].fillna("")
    fmask = df["fielder"] == fielder_name

    catches   = int(((df["dismissal_kind"].isin(["caught", "caught and bowled"])) & fmask).sum())
    run_outs  = int(((df["dismissal_kind"] == "run out") & fmask).sum())
    stumpings = int(((df["dismissal_kind"] == "stumped") & fmask).sum())

    # If you have a separate 'direct_hit' flag, use it here; else treat run‑outs as direct hits
    direct_hits = run_outs
    dismissals  = catches + run_outs + stumpings

    impact = catches * 8 + run_outs * 12 + stumpings * 10  # tweak weights freely

    return {
        "Catches": catches,
        "Run Outs": run_outs,
        "Stumpings": stumpings,
        "Direct Hits": direct_hits,
        "Dismissals Involved": dismissals,
        "Fielding Impact Score": impact,
    }


def highest_match_score(deliv_df: pd.DataFrame, matches_df: pd.DataFrame, batter: str):
    """
    Return a dictionary with the batter's highest IPL score
    (single match) plus contextual info.
    """
    bdf = deliv_df[deliv_df["batter"] == batter]
    if bdf.empty:
        raise ValueError(f"No deliveries found for batter '{batter}'")

    # Total runs per match_id
    match_totals = bdf.groupby("match_id")["batsman_runs"].sum()

    top_match_id = match_totals.idxmax()
    top_runs     = int(match_totals.loc[top_match_id])

    # Subset rows for that match only
    top_bdf = bdf[bdf["match_id"] == top_match_id]

    # Balls faced (exclude wides & no-balls)
    balls = int((~top_bdf["extras_type"].isin(["wides", "noballs"])).sum())
    sr    = round(top_runs / balls * 100, 2) if balls else np.nan

    # Boundaries
    fours = int((top_bdf["batsman_runs"] == 4).sum())
    sixes = int((top_bdf["batsman_runs"] == 6).sum())

    # Match metadata
    meta = matches_df[matches_df["id"] == top_match_id].iloc[0]
    date  = meta["date"]
    city  = meta["city"]
    venue = meta["venue"]
    batting_team = top_bdf["batting_team"].iloc[0]
    opponent = meta["team1"] if meta["team1"] != batting_team else meta["team2"]

    return {
        "Batter": batter,
        "Runs": top_runs,
        "Balls": balls,
        "Strike Rate": sr,
        "Fours": fours,
        "Sixes": sixes,
        "Match ID": int(top_match_id),
        "Date": date,
        "City": city,
        "Venue": venue,
        "Batting Team": batting_team,
        "Opponent": opponent,
    }





# Plotting Function

def player_stats_graph(player="V Kohli") -> None:
    bat_df  = pd.DataFrame([batsman_statistics(delivieres_dataset, player)])
    bowl_df = pd.DataFrame([bowler_statistics(delivieres_dataset, player)])
    fld_df  = pd.DataFrame([feilder_statistics(delivieres_dataset, player)])
    top_inn = highest_match_score(delivieres_dataset, matches_dataset, player)

    display(Markdown(f"<center><h1>{player} Performance Metrics</h1></center>"))
    display(Markdown(f"<center><h1>     </h1></center>"))

    display(bat_df, bowl_df, fld_df, pd.DataFrame([top_inn]))

    display(Markdown(f"<center><h1>     </h1></center>"))
    display(Markdown(f"<center><h1>{player} Performance Plots</h1></center>"))
    display(Markdown(f"<center><h1>     </h1></center>"))

    # Divergent bars  ➜  Runs vs Balls
    diff_runs = bat_df["Runs Scored"].iloc[0]
    diff_balls = bat_df["Balls Faced"].iloc[0]
    plt.figure(figsize=(18, 1))
    plt.barh([0], [-diff_balls] )
    plt.barh([0], [diff_runs])
    plt.text(0, 0, f"SR {bat_df['Strike Rate'].iloc[0]:.1f}",
             ha="center", va="center", color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc="#0a2642", ec="#0a2642"))
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.axvline(0, color="black")
    plt.title("Runs vs Balls (Divergent)",color='white')
    plt.tight_layout()
    plt.show()

    # Batting metric groups
    bar_plot(bat_df,
             exclude={"50s / 100s", "Dismissal Types", "Runs Scored",
                      "Balls Faced", "Fours", "Sixes", "Batting Impact Score"},
             title="Batting – Scoring")
    bar_plot(bat_df,
             exclude={"Runs Scored", "Balls Faced", "Strike Rate", "Batting Average",
                      "Dot Ball %", "50s / 100s", "Dismissal Types", "Consistency",
                      "Boundary %", "Runs per Match", "High Score"},
             title="Batting – Impact")

    # Bowling metric groups
    bar_plot(bowl_df,
             exclude={'Best Bowling Figures', 'Dot Ball %', 'Overs Bowled',
                      'Bowling Strike Rate', 'Runs Conceded', 'Bowling Average'},
             title="Bowling – Key Impact")
    bar_plot(bowl_df,
             exclude={"Wickets Taken", "Economy Rate", "Maiden Overs",
                      "Best Bowling Figures", "4-Wicket Hauls", "5-Wicket Hauls",
                      "Wide Balls", "No Balls"},
             title="Bowling – Economy & Hauls")

    # Pie chart – fielding
    labels = ["Catches", "Run Outs", "Stumpings", "Direct Hits"]
    sizes  = fld_df[labels].iloc[0].astype(float).tolist()
    plt.figure(figsize=(18, 10))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor="white"))
    plt.title(f"{player} – Fielding Dismissals",color='white')
    plt.tight_layout()
    plt.show()

    # KPI card from highest-scoring inning
    display(kpi_card(f"{player} – Highest IPL Innings", top_inn))


def kpi_card(title: str, kpis: dict) -> widgets.VBox:
    title_html = widgets.HTML(
        f"<b style='font-size:15px'>{title}</b>",
        layout=widgets.Layout(margin="0 0 8px 0")
    )
    rows = [
        widgets.HTML(
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<span>{k}</span><span><b>{v}</b></span></div>"
        )
        for k, v in kpis.items()
    ]
    return widgets.VBox(
        [title_html] + rows,
        layout=widgets.Layout(
            padding="10px 14px",
            border="1px solid #ccc",
            border_radius="10px",
            box_shadow="2px 2px 6px rgba(0,0,0,.15)",
            background_color="#f9fafb",
            width="230px"
        )
    )

def bar_plot(statistic_df: pd.DataFrame, exclude: set[str], title: str) -> None:
    row = statistic_df.iloc[0]
    metrics, scores = [], []

    for k, v in row.items():
        if k in exclude:
            continue
        try:
            scores.append(float(v))
            metrics.append(k)
        except ValueError:
            pass  # skip non-numeric

    plot_df = pd.DataFrame({"Metric": metrics, "Value": scores})
    plt.figure(figsize=(18, 5))
    sns.barplot(data=plot_df, y="Metric", x="Value", edgecolor="white", linewidth=0.8)
    plt.title(title,color='white')
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# usage
player_stats_graph()
```
## 🧠 Player Statistics Analysis Module – README

This module enables advanced IPL player performance analysis. It computes batting, bowling, and fielding metrics and visualizes them with plots. Below is a breakdown of each component, with each line explained:

---

### 🔸 Batsman Statistics

```python
def batsman_statistics(dataframe, batter_name, consistency_threshold=30):
```

* Defines a function to analyze batting performance of a specific player.

```python
    bf = dataframe[dataframe["batter"] == batter_name].copy()
```

* Filters the dataframe for only the rows where the player was the batter.

```python
    runs = int(bf["batsman_runs"].sum())
```

* Calculates total runs scored by summing the `batsman_runs` column.

```python
    legal_mask = ~bf["extras_type"].isin(["wides"])
```

* Creates a mask to exclude wides from legal deliveries.

```python
    balls_faced = int(legal_mask.sum())
```

* Calculates the total number of legal balls faced.

```python
    strike_rate = (runs / balls_faced * 100) if balls_faced else np.nan
```

* Computes strike rate; handles division by zero.

```python
    fours = int((bf["batsman_runs"] == 4).sum())
    sixes = int((bf["batsman_runs"] == 6).sum())
    boundary_runs = fours * 4 + sixes * 6
```

* Counts the number of 4s and 6s; calculates total runs from boundaries.

```python
    dot_balls = int((legal_mask & (bf["batsman_runs"] == 0)).sum())
    dot_ball_pct = (dot_balls / balls_faced * 100) if balls_faced else np.nan
```

* Calculates the number and percentage of dot balls.

```python
    match_runs = bf.groupby("match_id")["batsman_runs"].sum()
    matches_played = len(match_runs)
```

* Aggregates total runs per match and counts matches played.

```python
    fifties = int(match_runs[(match_runs >= 50) & (match_runs < 100)].count())
    hundreds = int(match_runs[match_runs >= 100].count())
    fifties_hundreds = f"{fifties} / {hundreds}"
```

* Calculates number of 50s and 100s.

```python
    high_score = int(match_runs.max()) if matches_played else np.nan
    runs_per_match = (runs / matches_played) if matches_played else np.nan
```

* Finds highest score and average runs per match.

```python
    consistency = (
        match_runs[match_runs >= consistency_threshold].count() / matches_played * 100
        if matches_played else np.nan
    )
```

* Calculates consistency based on the threshold.

```python
    outs_mask = (dataframe["is_wicket"] == 1) & (dataframe["player_dismissed"] == batter_name)
    outs = int(outs_mask.sum())
    batting_average = (runs / outs) if outs else np.nan
```

* Calculates how often the batter was out and computes batting average.

```python
    dismissal_types = (
        dataframe[outs_mask]["dismissal_kind"].value_counts().to_dict()
    )
```

* Gets count of dismissal types.

```python
    boundary_pct = (boundary_runs / runs * 100) if runs else np.nan
```

* Computes what percentage of runs came from boundaries.

```python
    batting_impact_score = runs_per_match * strike_rate if not np.isnan(runs_per_match) else np.nan
```

* A custom impact score: how productive the player is based on scoring rate and consistency.

Returns all metrics as a dictionary.

---

### 🔸 Bowler Statistics

```python
def bowler_statistics(df, bowler_name):
```

* Defines a function to analyze bowling stats.
* Filters delivery rows for the bowler.
* Excludes extras like wides and no-balls to calculate legal deliveries.
* Calculates overs bowled, wickets, economy, average, strike rate.
* Identifies best bowling figures, maidens, hauls (4W, 5W), and extras.
* Computes a custom bowling impact score.

---

### 🔸 Fielder Statistics

```python
def feilder_statistics(df, fielder_name):
```

* Calculates:

  * Number of catches, run-outs, stumpings.
  * Total dismissals involved.
  * Custom fielding impact score (weighted sum).

---

### 🔸 Highest Match Score

```python
def highest_match_score(deliv_df, matches_df, batter):
```

* Finds batter's best match (most runs scored).
* Calculates match strike rate, boundaries, and pulls match metadata (venue, date, opponent).

---

### 📊 Player Visualization

```python
def player_stats_graph(player="V Kohli"):
```

* Main function to display and plot:

  * Tables for batting, bowling, fielding, top match.
  * Graphs: Divergent bar (runs vs balls), grouped batting/bowling stats, pie chart (fielding).

---

### 🧩 Helper Functions

#### KPI Card

```python
def kpi_card(title, kpis):
```

* Creates a styled box to present key metrics.

#### Bar Plot

```python
def bar_plot(statistic_df, exclude, title):
```

* Plots a horizontal bar chart for selected statistics, excluding certain fields.

---

### 🚀 Usage

```python
player_stats_graph("MS Dhoni")
```

* Call with any player name to visualize their complete stats and plots.

---

> ⚠️ Note: This module assumes `delivieres_dataset` and `matches_dataset` are loaded.

```python
# Team Plot

# Function

def team_season_stats(deliv_df: pd.DataFrame, matches_df: pd.DataFrame, team: str):
    team_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)].copy()
    if team_matches.empty:
        raise ValueError(f"No matches found for team '{team}'")

    season_map = matches_df.set_index("id")["season"]
    deliv_df = deliv_df.copy()
    deliv_df["season"] = deliv_df["match_id"].map(season_map)

    stats = []

    for season, mdf in team_matches.groupby("season"):
        matches_played = len(mdf)
        wins = int((mdf["winner"] == team).sum())
        ties_no_result = int(((mdf["winner"].isna()) | (mdf["winner"] == "")).sum())
        losses = matches_played - wins - ties_no_result
        win_pct = wins / matches_played * 100 if matches_played else np.nan

        sdeliv = deliv_df[deliv_df["season"] == season]
        bat_df = sdeliv[sdeliv["batting_team"] == team]
        bowl_df = sdeliv[sdeliv["bowling_team"] == team]

        runs_for = int(bat_df["total_runs"].sum())
        runs_against = int(bowl_df["total_runs"].sum())
        legal_bat_balls = int((bat_df["extras_type"] != "wides").sum())
        legal_bowl_balls = int((bowl_df["extras_type"] != "wides").sum())

        overs_for = legal_bat_balls / 6 if legal_bat_balls else np.nan
        overs_against = legal_bowl_balls / 6 if legal_bowl_balls else np.nan

        nrr = (runs_for / overs_for - runs_against / overs_against) if overs_for and overs_against else np.nan

        stats.append({
            "Season": str(season),
            "Matches": matches_played,
            "Wins": wins,
            "Losses": losses,
            "WinPct": round(win_pct, 2),
            "RunsScored": runs_for,
            "RunsConceded": runs_against,
            "AvgRunsFor": round(runs_for / matches_played, 2),
            "AvgRunsAgainst": round(runs_against / matches_played, 2),
            "NetRunRate": round(nrr, 3) if not np.isnan(nrr) else np.nan,
        })

    return pd.DataFrame(stats).set_index("Season").sort_index()


# Plot Function


def Team_graph_plot(team_name="Mumbai Indians") -> None:
    data = team_season_stats(delivieres_dataset, matches_dataset, team_name)

    display(Markdown(f"<center><h1>{team_name} Performance Metric</h1></center>"))
    display(Markdown(f"<center><h1>     </h1></center>"))
    
    display(data)

    display(Markdown(f"<center><h1>{team_name} Performance Plots</h1></center>"))


    # Season is in index; move it to a column
    data = data.reset_index()


    # run scored vs run conceded
    # Melt the DataFrame
    df_long = data.melt(id_vars="Season",value_vars=["RunsScored", "RunsConceded"],var_name="Metric",value_name="Runs")

    # Plot
    plt.figure(figsize=(16, 4))
    sns.lineplot(data=df_long, x="Season", y="Runs", hue="Metric", marker="o", linewidth=2.2)

    plt.title(f"{team_name} – Runs Scored vs. Conceded by Season",color='white')
    plt.ylabel("Total Runs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # Avg run for vs Avg Runs Against
    # Melt the DataFrame
    data = data.reset_index()
    long_df = data.melt(id_vars="Season",
                        value_vars=["AvgRunsFor","AvgRunsAgainst"],
                        var_name="Metric",
                        value_name="Runs")

    # 3️⃣  bar plot (dodge=True ➜ side-by-side)
    plt.figure(figsize=(16, 5))
    sns.barplot(data=long_df, x="Season", y="Runs",
                hue="Metric", palette=["#eaa43c", "#d97030"], dodge=True)

    plt.title(f"{team_name} – Runs Scored vs. Conceded by Season",color='white')
    plt.ylabel("Total Runs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #season win pct

    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    sns.barplot( data=data, y="Season", x="WinPct", linewidth=1.2,ax=ax)

    ax.set_title("Win Percentage by Season",color='white')
    ax.set_xlabel("Win %")
    ax.set_ylabel("Season")
    ax.set_xlim(0, 100)     # WinPct is a percentage
    plt.tight_layout()
    plt.show()

    # Net run rate
    plt.figure(figsize=(16, 8))
    sns.barplot(data=data, x="Season", y="NetRunRate")   # one bar per category
    plt.title("NetRunRate",color='white')
    plt.ylabel("Season")
    plt.tight_layout()
    plt.show()

    # matches win /looses
    df = data
    vals_win  =  df["Wins"].values
    vals_loss = -df["Losses"].values
    seasons   =  df["Season"].astype(str).values

    y = range(len(df))           # y-axis positions

    fig, ax = plt.subplots(figsize=(16, 0.6*len(df)))

    # Plot losses (left) & wins (right)
    ax.barh(y, vals_loss)
    ax.barh(y, vals_win )

    # Season labels centred at x=0 with padding
    for y_pos, season in zip(y, seasons):
        ax.text(0, y_pos, season, ha="center", va="center",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#0a2642", ec="#0a2642"))

    # Formatting
    ax.set_yticks([])                 # hide y-axis
    ax.set_xlabel("Matches")
    ax.axvline(0, color="black")      # centre line
    ax.set_title("Wins vs Losses by Season (Divergent View)",color='white')
    plt.tight_layout()
    plt.show()

#usage
Team_graph_plot()
```
##

---

## 🟦 Team Season Analysis – Line‑by‑Line Guide

Below is a **line‑level walkthrough** of the two key team‑level helpers: `team_season_stats` (data prep) and `Team_graph_plot` (visualisation wrapper).

---

### 1️⃣ Function: `team_season_stats`

```python
def team_season_stats(deliv_df: pd.DataFrame, matches_df: pd.DataFrame, team: str):
```

*Defines a function that returns season‑wise metrics for a single IPL franchise.*

```python
    team_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)].copy()
```

* Filters `matches_df` for rows where the given team appears in either `team1` or `team2` and copies the slice so later edits won’t touch the original DataFrame.

```python
    if team_matches.empty:
        raise ValueError(f"No matches found for team '{team}'")
```

* Defensive check: if the filtered DataFrame is empty, the team string is invalid—raise a clear error.

```python
    season_map = matches_df.set_index("id")["season"]
```

* Builds a **Series** that maps each `match_id` (index) to its `season`. This will be used to stamp a season onto every ball in the deliveries table.

```python
    deliv_df = deliv_df.copy()
    deliv_df["season"] = deliv_df["match_id"].map(season_map)
```

* Copies deliveries DataFrame and appends a new `season` column by mapping its `match_id` through `season_map`.

```python
    stats = []
```

* Placeholder list; each iteration of the upcoming loop appends a dictionary of season metrics.

```python
    for season, mdf in team_matches.groupby("season"):
```

* Iterates over the team’s matches one **season** at a time.

```python
        matches_played = len(mdf)
```

* Counts fixtures for that season.

```python
        wins = int((mdf["winner"] == team).sum())
```

* Counts wins where `winner` matches `team`.

```python
        ties_no_result = int(((mdf["winner"].isna()) | (mdf["winner"] == "")).sum())
```

* Counts ties/no‑results: `winner` is either `NaN` or an empty string.

```python
        losses = matches_played - wins - ties_no_result
```

* Losses derived by subtraction.

```python
        win_pct = wins / matches_played * 100 if matches_played else np.nan
```

* Win % guarded against divide‑by‑zero.

```python
        sdeliv = deliv_df[deliv_df["season"] == season]
```

* Narrows deliveries to the current season.

```python
        bat_df = sdeliv[sdeliv["batting_team"] == team]
        bowl_df = sdeliv[sdeliv["bowling_team"] == team]
```

* Splits that season’s deliveries into batting & bowling frames.

```python
        runs_for     = int(bat_df["total_runs"].sum())
        runs_against = int(bowl_df["total_runs"].sum())
```

* Totals runs scored (“for”) and conceded (“against”).

```python
        legal_bat_balls  = int((bat_df["extras_type"] != "wides").sum())
        legal_bowl_balls = int((bowl_df["extras_type"] != "wides").sum())
```

* Counts legal deliveries (excluding wides) for both innings.

```python
        overs_for     = legal_bat_balls / 6 if legal_bat_balls else np.nan
        overs_against = legal_bowl_balls / 6 if legal_bowl_balls else np.nan
```

* Converts legal balls to overs (6 balls each). Uses `np.nan` if no balls.

```python
        nrr = (
            runs_for / overs_for - runs_against / overs_against
        ) if overs_for and overs_against else np.nan
```

* Computes **Net Run Rate** = Run Rate For − Run Rate Against.

```python
        stats.append({
            "Season": str(season),
            "Matches": matches_played,
            "Wins": wins,
            "Losses": losses,
            "WinPct": round(win_pct, 2),
            "RunsScored": runs_for,
            "RunsConceded": runs_against,
            "AvgRunsFor": round(runs_for / matches_played, 2),
            "AvgRunsAgainst": round(runs_against / matches_played, 2),
            "NetRunRate": round(nrr, 3) if not np.isnan(nrr) else np.nan,
        })
```

* Packages every metric into a dictionary and appends to `stats`.

```python
    return (
        pd.DataFrame(stats)         # → DataFrame of dicts
          .set_index("Season")     # index by Season
          .sort_index()            # chronological order
    )
```

* Converts list → DataFrame, indexes by Season, sorts, and returns.

---

### 2️⃣ Function: `Team_graph_plot`

```python
def Team_graph_plot(team_name="Mumbai Indians") -> None:
```

*High‑level function that prints the season table and renders five visualisations.*

```python
    data = team_season_stats(delivieres_dataset, matches_dataset, team_name)
```

* Calls the helper to compute season metrics for the chosen team.

```python
    display(Markdown(f"<center><h1>{team_name} Performance Metric</h1></center>"))
    display(data)
```

* Shows a centered heading in the notebook and prints the stats table.

```python
    data = data.reset_index()
```

* Moves `Season` out of the index so it behaves like a normal column for Seaborn.

#### 📈 Plot #1 — Runs Scored **vs** Runs Conceded

```python
    df_long = data.melt(...)
    sns.lineplot(...)
```

* Melts wide → long form so two metrics plot as separate lines with markers.

#### 📊 Plot #2 — Average Runs (Side‑by‑Side Bars)

```python
    long_df = data.melt(...)
    sns.barplot(..., dodge=True)
```

* Draws grouped bars (`AvgRunsFor` vs `AvgRunsAgainst`) per season.

#### 📊 Plot #3 — Win Percentage

```python
    sns.barplot(y="Season", x="WinPct", ...)
```

* Horizontal bars make it easy to compare 0–100% scale.

#### 📊 Plot #4 — Net Run Rate by Season

```python
    sns.barplot(x="Season", y="NetRunRate")
```

* Single‑series vertical bar chart for quick trend reading.

#### 📊 Plot #5 — Wins vs Losses (Divergent Horizontal Bars)

```python
    vals_win  =  df["Wins"].values
    vals_loss = -df["Losses"].values
    ax.barh(...)
```

* Positive bars → wins (right), negative bars → losses (left).
* Adds season labels at x = 0 for symmetry.

```python
Team_graph_plot()
```

* Default call visualises Mumbai Indians. Pass any valid team string to change.

---

> 📌 These functions rely on previously loaded `delivieres_dataset` and `matches_dataset`. Ensure they are globally available before calling.

```python
# Orange Cap Purple Cap Table

# Function


def orange_cap_table(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame listing the IPL Orange-Cap winner (most runs) for every season.

    Columns returned:
      Season • Player • Runs
    """
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")

    # Attach season to each delivery
    season_map = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_map)

    # Season-wise run totals
    season_runs = (
        d.groupby(["season", "batter"])["batsman_runs"].sum()
          .reset_index()
    )

    # Pick top run-scorer per season
    orange = (
        season_runs.sort_values(["season", "batsman_runs"], ascending=[True, False])
                   .groupby("season").head(1)      # one row per season
                   .rename(columns={"batter": "Player", "batsman_runs": "Runs"})
                   .reset_index(drop=True)
    )

    return orange[["season", "Player", "Runs"]].rename(columns={"season": "Season"})


# ──────────────────────────────────────────────────────────
#  PURPLE-CAP TABLE  (highest wicket-taker per season)
# ──────────────────────────────────────────────────────────
def purple_cap_table(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame listing the IPL Purple-Cap winner (most wickets) for every season.

    Columns returned:
      Season • Player • Wickets
    """
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")

    season_map = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_map)

    # Only wickets credited to the bowler
    credited_kinds = {
        "bowled", "caught", "caught and bowled", "lbw",
        "stumped", "hit wicket", "hit-wicket", "caught & bowled"
    }
    wkt_rows = d[(d["is_wicket"] == 1) & (d["dismissal_kind"].isin(credited_kinds))]

    season_wkts = (
        wkt_rows.groupby(["season", "bowler"]).size()
                .reset_index(name="Wickets")
    )

    purple = (
        season_wkts.sort_values(["season", "Wickets"], ascending=[True, False])
                   .groupby("season").head(1)
                   .rename(columns={"bowler": "Player"})
                   .reset_index(drop=True)
    )

    return purple[["season", "Player", "Wickets"]].rename(columns={"season": "Season"})




# Function Plot
def cap_plot():

  display(Markdown(f"<center><h1>Orange Cap Metric</h1></center>"))
  display(Markdown(f"<center><h1></h1></center>"))


  orange_cap = orange_cap_table(delivieres_dataset,matches_dataset)
  display(orange_cap)

  display(Markdown(f"<center><h1></h1></center>"))
  display(Markdown(f"<center><h1>Purple Cap Metric</h1></center>"))
  display(Markdown(f"<center><h1></h1></center>"))


  purple_cap = purple_cap_table(delivieres_dataset,matches_dataset)
  display(purple_cap)

  display(Markdown(f"<center><h1>Orange Cap Plots</h1></center>"))
  display(Markdown(f"<center><h1></h1></center>"))
  # Plot Orange Cap
  # Plotting
  plt.figure(figsize=(18, 10))
  sns.lineplot(data=orange_cap, x="Season", y="Runs", hue="Player", marker="o", linewidth=2.2)

  plt.title("Orange Cap - Highest Run Scorer of the Season",color='white')
  plt.xlabel("Season")
  plt.ylabel("Total Runs")
  plt.xticks(orange_cap["Season"].unique())  # Ensure proper season ticks
  plt.grid(True)
  plt.tight_layout()
  plt.show()


  # Plotting Purple Cap
  display(Markdown(f"<center><h1>Purple Cap Plots</h1></center>"))
  display(Markdown(f"<center><h1></h1></center>"))

  plt.figure(figsize=(18, 10))
  sns.lineplot(data=purple_cap, x="Season", y="Wickets", hue="Player", marker="o", linewidth=2.2)

  plt.title("Purple Cap - Highest Wicket Taker Of the Season",color='white')
  plt.xlabel("Season")
  plt.ylabel("Total Wickets")
  plt.xticks(purple_cap["Season"].unique())  # Ensure proper season ticks
  plt.grid(True)
  plt.tight_layout()
  plt.show()

# usage
cap_plot()
```

```python
# Overall Stat Plot


def overall_batsman(deliveries_df: pd.DataFrame,
                    matches_df: pd.DataFrame,
                    season: str | None = "all") -> pd.DataFrame:
    """
    Return a DataFrame of the TOP-10 run-scorers.

    Parameters
    ----------
    deliveries_df : DataFrame   (IPL deliveries.csv)
    matches_df    : DataFrame   (IPL matches.csv)
    season        : str or None
        • "all" / "all season" / None  → overall (all seasons combined)
        • "2013", "2009/10", …         → that specific season only

    Columns returned
    ----------------
      Player • Runs • Balls • StrikeRate
    """
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")

    # Attach season to each delivery
    season_lookup = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_lookup)

    # Optional season filter
    if season and str(season).lower() not in {"all", "all season"}:
        d = d[d["season"] == season]

    # Aggregate
    legal_mask = ~d["extras_type"].isin(["wides", "noballs"])
    agg = (
        d.assign(legal_ball=legal_mask.astype(int))
         .groupby("batter")
         .agg(Runs=("batsman_runs", "sum"),
              Balls=("legal_ball", "sum"))
         .reset_index()
         .rename(columns={"batter": "Player"})
    )
    agg["StrikeRate"] = (agg["Runs"] / agg["Balls"] * 100).round(2)

    # Top-10
    top10 = agg.sort_values("Runs", ascending=False).head(10).reset_index(drop=True)
    return top10



def overall_bowler(deliveries_df: pd.DataFrame,
                   matches_df: pd.DataFrame,
                   season: str | None = "all") -> pd.DataFrame:
    """
    Return a DataFrame of the TOP-10 wicket-takers.

    Parameters
    ----------
    deliveries_df : DataFrame   (IPL deliveries.csv)
    matches_df    : DataFrame   (IPL matches.csv)
    season        : str or None
        • "all" / "all season" / None  → overall (all seasons combined)
        • "2013", "2009/10", …         → that specific season only

    Columns returned
    ----------------
      Player • Wickets • Overs • RunsConceded • Economy • Average • StrikeRate
    """
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")

    # Map season to each delivery
    season_lookup = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_lookup)

    # Optional season filter
    if season and str(season).lower() not in {"all", "all season"}:
        d = d[d["season"] == season]

    # Only wickets credited to bowler
    credited = {
        "bowled", "caught", "caught and bowled", "lbw",
        "stumped", "hit wicket", "hit-wicket", "caught & bowled"
    }
    wkt_rows = d[(d["is_wicket"] == 1) & (d["dismissal_kind"].isin(credited))]

    # Aggregate base metrics
    legal_mask = ~d["extras_type"].isin(["wides", "noballs"])
    bowl_base = (
        d.assign(legal_ball=legal_mask.astype(int))
          .groupby("bowler")
          .agg(RunsConceded=("total_runs", "sum"),
               LegalBalls=("legal_ball", "sum"))
    )

    wickets = wkt_rows.groupby("bowler").size().rename("Wickets")
    agg = bowl_base.join(wickets, how="left").fillna({"Wickets": 0})

    # Derived metrics
    agg["Overs"] = agg["LegalBalls"] / 6
    agg["Economy"] = (agg["RunsConceded"] / agg["Overs"]).round(2)
    agg["Average"] = (agg["RunsConceded"] / agg["Wickets"]).replace([np.inf, np.nan], 0).round(2)
    agg["StrikeRate"] = (agg["LegalBalls"] / agg["Wickets"]).replace([np.inf, np.nan], 0).round(2)

    agg = (
        agg.reset_index()
           .rename(columns={"bowler": "Player"})
           .astype({"Wickets": "int", "RunsConceded": "int"})
    )

    top10 = agg.sort_values("Wickets", ascending=False).head(10).reset_index(drop=True)
    return top10[["Player", "Wickets", "Overs", "RunsConceded", "Economy", "Average", "StrikeRate"]]






# overall player stat plot




def overall_stat_plot(Season = None):
    if Season == None:
      Season = "All"
    title = f"Top Batsman As Per {Season} Season"
    overall_batsman_df = overall_batsman(delivieres_dataset,matches_dataset,Season)

    display(Markdown(f"<center><h1>{title} Metrics</h1></center>"))
    display(Markdown(f"<center><h1></h1></center>"))
    display(overall_batsman_df)


    df = overall_bowler(delivieres_dataset,matches_dataset,Season)

    display(Markdown(f"<center><h1>Top Bowler As Per {Season} Season Metric</h1></center>"))
    display(Markdown(f"<center><h1></h1></center>"))
    display(df)


    display(Markdown(f"<center><h1>{title} Plots</h1></center>"))

    #
    x      = np.arange(len(overall_batsman_df))        # one slot per player
    width  = 0.25                      # width of each small bar

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.bar(x - width, overall_batsman_df["Runs"],    width, label="Runs Scored",    color="#2196f3")
    ax.bar(x , overall_batsman_df["Balls"], width, label="Balls",color="#ff9800")

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(overall_batsman_df["Player"])
    ax.set_ylabel("Value")
    ax.set_title("Bowler Metrics by Player",color='white')
    ax.legend()

    plt.tight_layout()
    plt.figure(figsize=(18, 10))
    plt.show()

    # Bar Plot Strike rate
    plt.figure(figsize=(18, 10))
    sns.barplot(overall_batsman_df, y="Player", x="StrikeRate", edgecolor="black")

    plt.title(title,color='white')
    plt.xlabel("StrikeRate")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()



    display(Markdown(f"<center><h1>Top Bowler As Per {Season} Season Plots</h1></center>"))

    # Bowler Plot

    title = f"Top Bowler As Per {Season} Season"

    x      = np.arange(len(df))        # one slot per player
    width  = 0.25                    # width of each small bar

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.bar(x - width, df["Economy"],    width, label="Economy",    color="#4caf50")
    ax.bar(x,          df["Average"],   width, label="Average",    color="#2196f3")
    ax.bar(x + width, df["StrikeRate"], width, label="Strike-Rate",color="#ff9800")

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(df["Player"])
    ax.set_ylabel("Value")
    ax.set_title("Bowler Metrics by Player",color='white')
    ax.legend()

    plt.tight_layout()
    plt.show()


  # usage
  # divergent plot
    vals_win  =  df["RunsConceded"].values
    vals_loss = -df["Overs"].values
    seasons   =  df["Player"].astype(str).values

    y = range(len(df))           # y-axis positions

    fig, ax = plt.subplots(figsize=(18, 0.6*len(df)))

    # Plot losses (left) & wins (right)
    ax.barh(y, vals_loss)
    ax.barh(y, vals_win )

    # Season labels centred at x=0 with padding
    for y_pos, season in zip(y, seasons):
        ax.text(0, y_pos, season, ha="center", va="center",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#0a2642", ec="#0a2642"))

    # Formatting
    ax.set_yticks([])                 # hide y-axis
    ax.set_xlabel("Overs Vs Run Conceded")
    ax.axvline(0, color="black")      # centre line
    ax.set_title("Wins vs Losses by Season (Divergent View)",color='white')
    plt.tight_layout()
    plt.show()

  # Bar Plot
    plt.figure(figsize=(18, 10))
    sns.barplot(df, y="Player", x="Wickets", edgecolor="black")

    plt.title(title,color='white')
    plt.xlabel("Wickets")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()





#usage
overall_stat_plot()

```

##

---

##

---

## 🟪 Orange & Purple Cap Analysis – Line‑by‑Line Guide

### 1️⃣ Function: `orange_cap_table`

```python
def orange_cap_table(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
```

*Returns the top run‑scorer (Orange Cap) for every IPL season.*

```python
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")
```

* Copies the deliveries frame and replaces `NaN` in `extras_type` with an empty string so later filters are string‑safe.

```python
    season_map = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_map)
```

* Builds a `match_id → season` map and stamps each delivery with its season.

```python
    season_runs = (
        d.groupby(["season", "batter"])["batsman_runs"].sum()
          .reset_index()
    )
```

* Aggregates total runs **per season per batter**.

```python
    orange = (
        season_runs.sort_values(["season", "batsman_runs"], ascending=[True, False])
                   .groupby("season").head(1)      # top row = highest scorer
                   .rename(columns={"batter": "Player", "batsman_runs": "Runs"})
                   .reset_index(drop=True)
    )
```

* Sorts each season by runs desc, keeps the first row (Orange Cap), renames columns.

```python
    return orange[["season", "Player", "Runs"]].rename(columns={"season": "Season"})
```

* Returns a tidy 3‑column DataFrame.

---

### 2️⃣ Function: `purple_cap_table`

```python
def purple_cap_table(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
```

*Returns the top wicket‑taker (Purple Cap) for every IPL season.*

```python
    d = deliveries_df.copy()
    d["extras_type"] = d["extras_type"].fillna("")
```

* Same cleaning step as before.

```python
    season_map = matches_df.set_index("id")["season"]
    d["season"] = d["match_id"].map(season_map)
```

* Adds `season` to each delivery.

```python
    credited_kinds = { ... }
    wkt_rows = d[(d["is_wicket"] == 1) & (d["dismissal_kind"].isin(credited_kinds))]
```

* Filters only wickets that count toward the bowler’s tally (no run‑outs etc.).

```python
    season_wkts = (
        wkt_rows.groupby(["season", "bowler"]).size()
                .reset_index(name="Wickets")
    )
```

* Counts wickets per bowler per season.

```python
    purple = (
        season_wkts.sort_values(["season", "Wickets"], ascending=[True, False])
                   .groupby("season").head(1)
                   .rename(columns={"bowler": "Player"})
                   .reset_index(drop=True)
    )
```

* Takes the first row (most wickets) per season.

```python
    return purple[["season", "Player", "Wickets"]].rename(columns={"season": "Season"})
```

* Final tidy table.

---

### 3️⃣ Function: `cap_plot`

High‑level routine that *displays tables* for Orange & Purple Cap and plots two line charts.

1. **Display headings** using `Markdown`.
2. **Compute tables**:

   ```python
   orange_cap = orange_cap_table(...)
   purple_cap = purple_cap_table(...)
   ```
3. **Show tables** with `display()`.
4. **Plot Orange Cap runs** – line plot with `Season` on x‑axis and `Runs` on y‑axis.
5. **Plot Purple Cap wickets** similarly.

All styling (figure size, colors, titles) is set inline.

---

## 🟥 Overall Season‑Top Players – Line‑by‑Line Guide

### 4️⃣ Function: `overall_batsman`

```python
def overall_batsman(deliveries_df, matches_df, season: str | None = "all") -> pd.DataFrame:
```

*Computes top‑10 run‑scorers either overall or for a specific season.*

* **Season stamping** – same `season_lookup` technique.
* **Optional filter** – skip if `season` is "all".
* **Aggregate** legal balls & runs per batter.
* **Strike Rate** computed then top‑10 sorted by `Runs`.

### 5️⃣ Function: `overall_bowler`

Similar structure but for bowlers, deriving:

* `Wickets`, `Overs`, `RunsConceded`, `Economy`, `Average`, `StrikeRate`.
* Uses `credited` dismissal kinds for accuracy.

### 6️⃣ Function: `overall_stat_plot`

1. **Prepare title** based on `Season` arg.
2. **Compute top tables** with `overall_batsman` & `overall_bowler`.
3. **Display tables** with Markdown headings.
4. **Batsman plots**

   * Grouped bar: `Runs` vs `Balls` per player.
   * Bar: `StrikeRate` per player.
5. **Bowler plots**

   * Grouped bar: `Economy`, `Average`, `StrikeRate`.
   * Divergent bar: `RunsConceded` (positive) vs `Overs` (negative) to visualise workload vs runs.
   * Bar: `Wickets` per player.
6. **Layout/tight\_layout()** calls ensure plots don’t overlap.

Call:

```python
overall_stat_plot()          # entire dataset
overall_stat_plot("2016")   # specific season
```

These utilities round out player‑level and season‑level leaderboards for your IPL analysis.
 
 ```python

# ─────────────────────────────────────────────────────────────
def toss_insight(matches_df: pd.DataFrame,
                 season: str | None = None,
                 team:   str | None = None) -> dict:
    """
    Return a dictionary of DataFrames that describe:
        • toss‑winner frequency
        • decision split (bat / field)
        • how often the toss winner also won the match

    Parameters
    ----------
    matches_df : the IPL matches.csv as a DataFrame.
    season     :  '2020', '2013', …   or None for all seasons.
    team       :  'KKR', 'MI', …      or None for all teams.

    Returns
    -------
    dict  {
        "freq":       toss‑win counts per team,
        "decision":   bat vs field counts,
        "match_outcome":  cross‑tab (tossWin & matchWin),
        "summary":    one‑row KPI table
    }
    """

    # 1️⃣  optional filters
    m = matches_df.copy()

    if season not in {None, "", "All", "all"}:
        m = m[m["season"] == season]

    if team not in {None, "", "All", "all"}:
        m = m[(m["team1"] == team) | (m["team2"] == team)]

    # 2️⃣  who wins the toss most?
    freq = (
        m["toss_winner"]
        .value_counts()
        .rename_axis("Team")
        .rename("Toss Wins")
        .reset_index()
    )

    # 3️⃣  decision split  (bat / field) for the chosen slice
    decision = (
        m.groupby("toss_decision")
         .size()
         .rename("Count")
         .reset_index()
         .rename(columns={"toss_decision": "Decision"})
    )

    # 4️⃣  did toss winner also win match?
    toss_match = pd.crosstab(
        m["toss_winner"] == m["winner"],
        m["toss_decision"],
        rownames=["TossWinner==MatchWinner?"], colnames=["Decision"],
        normalize="columns"  # fraction inside each decision column
    ).round(3)

    # 5️⃣  simple KPIs
    total_matches = len(m)
    toss_first = decision.loc[decision["Decision"] == "field", "Count"].sum()
    overall_advantage = (m["toss_winner"] == m["winner"]).mean()

    summary = pd.DataFrame({
        "Matches Analysed": [total_matches],
        "Field First (%)": [toss_first / total_matches * 100 if total_matches else 0],
        "Toss Winner Won Match (%)": [overall_advantage * 100 if total_matches else 0],
    })

    return {
        "freq": freq,
        "decision": decision,
        "match_outcome": toss_match,
        "summary": summary
    }

"""
# whole‑history insight
insight_all = toss_insight(matches_dataset)
display(insight_all["summary"])
display(insight_all["decision"])
"""
print("*"*50)

# franchise‑specific
#insight_kkr = toss_insight(matches_dataset, team="Kolkata Knight Riders",season="2009")["summary"]
#display(insight_kkr)

for teams in list_of_team :
  print(f"Toss Insight For {teams}")
  insight = toss_insight(matches_dataset, team=teams)["summary"]

  display(Markdown(f"<center><h1>Toss Insight Metrics</h1></center>"))
  display(insight)
  display(Markdown(f"<center><h1>Toss Insight Plots</h1></center>"))



def divergent_all_teams(matches_dataset,
                        list_of_team,
                        palette=("#e76f51", "#2a9d8f"),
                        figsize=(18, None)):
    """
    One divergent chart for all teams:
        • left  bar  = Field First (%)
        • right bar  = Toss‑Winner Won Match (%)
        • team label = centred at x = 0
        • matches    = text just left of bars

    Parameters
    ----------
    matches_dataset : your full IPL matches DataFrame
    list_of_team    : iterable of team names to include
    palette         : two‑colour tuple (left, right)
    figsize         : (width, height); height auto‑scales if None
    """

    # ── 1. Aggregate summaries for every team ───────────────────────────────
    frames = []
    for team in list_of_team:
        # Most toss_insight implementations return a single‑row df in ["summary"]
        # Adjust `.iloc[0]` etc. if your function returns something else.
        summary_df = toss_insight(matches_dataset, team=team)["summary"].copy()
        summary_df["Team"] = team                         # keep team id
        frames.append(summary_df)

    if not frames:
        raise ValueError("No team data collected – check team names.")

    df_all = pd.concat(frames, ignore_index=True)

    # Ensure expected columns exist
    required = {"Team", "Matches Analysed",
                "Field First (%)", "Toss Winner Won Match (%)"}
    missing = required - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing columns in summary: {missing}")

    # ── 2. Prep for plotting ────────────────────────────────────────────────
    df_all["Field (-)"] = -df_all["Field First (%)"]       # mirror left side

    # Auto‑height ≈ 0.6 inch per row if not supplied
    if figsize[1] is None:
        figsize = (figsize[0], 0.6 * len(df_all))

    fig, ax = plt.subplots(figsize=figsize)

    # Left (Field First) & Right (Toss Winner) bars
    ax.barh(df_all["Team"], df_all["Field (-)"],
            color=palette[0], label="Field First (%)")
    ax.barh(df_all["Team"], df_all["Toss Winner Won Match (%)"],
            color=palette[1], label="Toss Winner Won Match (%)")

    # ── 3. Decorations ──────────────────────────────────────────────────────
    ax.axvline(0, color="black", lw=1)                     # centre line
    ax.set_yticks([])                                      # hide default ticks

    # Team label centred at x = 0
    for y_pos, team in enumerate(df_all["Team"]):
        ax.text(0, y_pos, team,
                ha="center", va="center", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#0a2642", ec="#0a2642"))

    # Total matches outside left margin
    x_left = ax.get_xlim()[0]
    for y_pos, matches in enumerate(df_all["Matches Analysed"]):
        ax.text(x_left - 3, y_pos, f"{matches} matches",
                ha="right", va="center",color = "white", fontsize=8)

    # Axis / legend / title
    ax.set_xlabel("Percentage")
    ax.set_title("Field‑First vs Toss‑Winner Success — All Teams", color = "white",pad=14)
    ax.xaxis.set_major_formatter(lambda x, _: f"{abs(x):.0f}%")
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.show()



# usage
divergent_all_teams(matches_dataset, list_of_team)

```

## Toss Insight Function and Plot – Line by Line Explanation

### `toss_insight()`

```python
def toss_insight(matches_df: pd.DataFrame, season: str | None = None, team: str | None = None) -> dict:
```

* Define a function to compute toss statistics.
* Parameters: `matches_df` (DataFrame of match data), optional `season`, optional `team`.

```python
    m = matches_df.copy()
```

* Copy the match dataset for safety (avoid modifying original).

```python
    if season not in {None, "", "All", "all"}:
        m = m[m["season"] == season]
```

* If a specific season is given, filter data for that season only.

```python
    if team not in {None, "", "All", "all"}:
        m = m[(m["team1"] == team) | (m["team2"] == team)]
```

* If a specific team is provided, filter matches where the team played.

#### Toss Win Frequency

```python
    freq = (
        m["toss_winner"]
        .value_counts()
        .rename_axis("Team")
        .rename("Toss Wins")
        .reset_index()
    )
```

* Count how many times each team won the toss.
* Return as DataFrame with columns: `Team`, `Toss Wins`.

#### Toss Decision Split

```python
    decision = (
        m.groupby("toss_decision")
         .size()
         .rename("Count")
         .reset_index()
         .rename(columns={"toss_decision": "Decision"})
    )
```

* Count how many times toss winners chose to bat or field.

#### Toss Winner vs Match Winner

```python
    toss_match = pd.crosstab(
        m["toss_winner"] == m["winner"],
        m["toss_decision"],
        rownames=["TossWinner==MatchWinner?"], colnames=["Decision"],
        normalize="columns"
    ).round(3)
```

* Check correlation between toss win and match win.
* Normalized so values are % within each decision.

#### Summary Metrics

```python
    total_matches = len(m)
    toss_first = decision.loc[decision["Decision"] == "field", "Count"].sum()
    overall_advantage = (m["toss_winner"] == m["winner"]).mean()
```

* Compute number of matches, field-first count, and overall win rate for toss winners.

```python
    summary = pd.DataFrame({
        "Matches Analysed": [total_matches],
        "Field First (%)": [toss_first / total_matches * 100 if total_matches else 0],
        "Toss Winner Won Match (%)": [overall_advantage * 100 if total_matches else 0],
    })
```

* Create a one-row KPI summary DataFrame.

#### Return All Tables

```python
    return {
        "freq": freq,
        "decision": decision,
        "match_outcome": toss_match,
        "summary": summary
    }
```

* Return a dictionary of all four result tables.

---

## `divergent_all_teams()`

```python
def divergent_all_teams(matches_dataset, list_of_team, palette=("#e76f51", "#2a9d8f"), figsize=(18, None)):
```

* Creates a divergent bar chart comparing each team's toss decision and success.

```python
    frames = []
    for team in list_of_team:
        summary_df = toss_insight(matches_dataset, team=team)["summary"].copy()
        summary_df["Team"] = team
        frames.append(summary_df)
```

* For each team, get the toss summary and add a team column.

```python
    if not frames:
        raise ValueError("No team data collected – check team names.")
```

* Handle case where list is empty or data missing.

```python
    df_all = pd.concat(frames, ignore_index=True)
```

* Combine all team summaries into one DataFrame.

```python
    required = {"Team", "Matches Analysed", "Field First (%)", "Toss Winner Won Match (%)"}
    missing = required - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing columns in summary: {missing}")
```

* Ensure required columns exist before plotting.

```python
    df_all["Field (-)"] = -df_all["Field First (%)"]
```

* Mirror field-first data for divergent plot.

```python
    if figsize[1] is None:
        figsize = (figsize[0], 0.6 * len(df_all))
```

* Auto scale height if not provided.

#### Plot Bars

```python
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df_all["Team"], df_all["Field (-)"], color=palette[0], label="Field First (%)")
    ax.barh(df_all["Team"], df_all["Toss Winner Won Match (%)"], color=palette[1], label="Toss Winner Won Match (%)")
```

* Plot horizontal bars for both stats.

#### Decorations

```python
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([])
```

* Add center line and remove default y-axis ticks.

```python
    for y_pos, team in enumerate(df_all["Team"]):
        ax.text(0, y_pos, team, ha="center", va="center", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0a2642", ec="#0a2642"))
```

* Place team name at center.

```python
    x_left = ax.get_xlim()[0]
    for y_pos, matches in enumerate(df_all["Matches Analysed"]):
        ax.text(x_left - 3, y_pos, f"{matches} matches", ha="right", va="center", color="white", fontsize=8)
```

* Show match count on left of chart.

```python
    ax.set_xlabel("Percentage")
    ax.set_title("Field‑First vs Toss‑Winner Success — All Teams", color="white", pad=14)
    ax.xaxis.set_major_formatter(lambda x, _: f"{abs(x):.0f}%")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()
```

* Set labels, format as percentage, show legend and render plot.

---

## Example Usage:

```python
insight_all = toss_insight(matches_dataset)
display(insight_all["summary"])
display(insight_all["decision"])
```

* Show overall toss metrics.

```python
for teams in list_of_team:
    print(f"Toss Insight For {teams}")
    insight = toss_insight(matches_dataset, team=teams)["summary"]
    display(Markdown(f"<center><h1>Toss Insight Metrics</h1></center>"))
    display(insight)
    display(Markdown(f"<center><h1>Toss Insight Plots</h1></center>"))
```

* Loop over each team and display their toss performance.

```python
divergent_all_teams(matches_dataset, list_of_team)
```

* Generate the comparative divergent chart across all teams.

```python
def corrected_venue_insights(matches_df, deliveries_df):
    merged_df = deliveries_df.merge(matches_df[['id', 'venue']], left_on='match_id', right_on='id', how='left')

    first_innings = merged_df[merged_df['inning'] == 1]
    first_innings_total = first_innings.groupby(['match_id', 'venue'])['total_runs'].sum().reset_index()
    avg_1st_innings_score = first_innings_total.groupby('venue')['total_runs'].mean()

    matches_df['batting_first_win'] = matches_df.apply(
        lambda x: 1 if x['toss_decision'] == 'field' and x['winner'] != x['toss_winner']
        else (1 if x['toss_decision'] == 'bat' and x['winner'] == x['toss_winner'] else 0),
        axis=1
    )
    matches_df['chasing_win'] = 1 - matches_df['batting_first_win']

    match_counts = matches_df['venue'].value_counts()
    win_type_counts = matches_df.groupby('venue')[['batting_first_win', 'chasing_win']].sum()

    summary = pd.DataFrame({
        'Matches': match_counts,
        'Wins Batting First': win_type_counts['batting_first_win'],
        'Wins Chasing': win_type_counts['chasing_win'],
        'Avg 1st Inn Score': avg_1st_innings_score
    }).fillna(0).astype({'Matches': int, 'Wins Batting First': int, 'Wins Chasing': int})

    summary['Bat First Win %'] = (summary['Wins Batting First'] / summary['Matches']) * 100
    summary['Chase Win %'] = (summary['Wins Chasing'] / summary['Matches']) * 100

    return summary.reset_index().rename(columns={'index': 'Venue'}).sort_values('Matches', ascending=False)


def plot_venue_insight():


    # Load and prepare data
    df = corrected_venue_insights(matches_dataset, delivieres_dataset)

    display(Markdown(f"<center><h1>Venue Insight Metrics</h1></center>"))
    display(df)

    # Convert relevant columns to numeric
    num_cols = ['Matches', 'Wins Batting First', 'Wins Chasing',
                'Avg 1st Inn Score', 'Bat First Win %', 'Chase Win %']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])

    # Filter top venues
    df_bar = df.sort_values('Matches', ascending=False).head(10).reset_index(drop=True)
    df_score = df.sort_values('Avg 1st Inn Score', ascending=False).head(15).reset_index(drop=True)
    df_pct = df_bar.copy()


    display(Markdown(f"<center><h1>Venue Insight Plots</h1></center>"))

    # 1. Stacked Bar Chart: Wins by batting order
    # ── assume df_bar already contains only the 10 venues you want ──
    df_bar = df_bar.copy()
    df_bar["Total Wins"] = df_bar["Wins Batting First"] + df_bar["Wins Chasing"]
    df_bar = df_bar.sort_values("Total Wins")          # smallest at top → invert later

    plt.figure(figsize=(10, 7))
    # first layer
    plt.barh(
        y=df_bar["venue"],
        width=df_bar["Wins Batting First"],
        color="orange",
        label="Bat First Wins",
    )
    # stacked layer (use left=… instead of bottom=…)
    plt.barh(
        y=df_bar["venue"],
        width=df_bar["Wins Chasing"],
        left=df_bar["Wins Batting First"],
        color="green",
        label="Chase Wins",
    )

    # cosmetics
    plt.xlabel("Total Wins", color="white", fontsize=12)
    plt.title("Wins by Batting Order – Top 10 Venues", fontsize=14, color="white")
    plt.gca().invert_yaxis()            # biggest bar on top
    plt.legend()
    plt.tight_layout()
    plt.show()


    # 2. Horizontal Bar Chart: Average 1st Innings Score
    plt.figure(figsize=(10, 8))
    sns.barplot(
        y="venue", x="Avg 1st Inn Score",
        data=df_score.sort_values('Avg 1st Inn Score'),
        palette="Blues_d")
    plt.title("Average 1st Innings Score – Top 15 Venues", fontsize=14, color='white')
    plt.xlabel("Avg Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # 3. Scatter Plot: Matches vs Avg Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Matches", y="Avg 1st Inn Score",
        data=df, s=80, edgecolor="black", color="purple")
    plt.title("Matches vs. Avg 1st Innings Score", fontsize=14, color='white')
    plt.xlabel("Matches Played")
    plt.ylabel("Avg 1st Inn Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4. Grouped Bar Chart: Bat First Win % vs Chase Win %
    df_pct = df_pct.sort_values("venue")          # or any other key

    y = np.arange(len(df_pct))           # one slot per venue
    h = 0.35                             # bar “thickness” (like width in bar())

    plt.figure(figsize=(10, 7))

    # Bat‑first bar (upper half of the slot)
    plt.barh(
        y - h/2,
        df_pct["Bat First Win %"],
        height=h,
        label="Bat First %",
        color="coral",
    )

    # Chase bar (lower half of the slot)
    plt.barh(
        y + h/2,
        df_pct["Chase Win %"],
        height=h,
        label="Chase %",
        color="mediumseagreen",
    )

    # y‑axis ticks & labels
    plt.yticks(y, df_pct["venue"])
    plt.gca().invert_yaxis()             # largest y (top of df) appears first

    # axis labels & title
    plt.xlabel("Win Percentage")
    plt.title("Win Percentage – Bat First vs Chase", fontsize=14, color="white")
    plt.legend()

    plt.tight_layout()
    plt.show()

# usage
plot_venue_insight()
```

##

---

## 🏟️ Venue Insight – Line‑by‑Line Guide

### 1️⃣ Function `corrected_venue_insights()`

```python
def corrected_venue_insights(matches_df, deliveries_df):
```

*Computes match counts, win splits, and average first‑innings scores for every IPL venue.*

```python
    merged_df = deliveries_df.merge(matches_df[['id', 'venue']], left_on='match_id', right_on='id', how='left')
```

* Join deliveries with venue names (from `matches_df`) on `match_id` → new DataFrame `merged_df`.

```python
    first_innings = merged_df[merged_df['inning'] == 1]
```

* Keep only first‑innings deliveries.

```python
    first_innings_total = first_innings.groupby(['match_id', 'venue'])['total_runs'].sum().reset_index()
```

* Sum runs for each match’s first innings.

```python
    avg_1st_innings_score = first_innings_total.groupby('venue')['total_runs'].mean()
```

* Average first‑innings total per venue.

#### Flag winner by strategy

```python
    matches_df['batting_first_win'] = matches_df.apply(
        lambda x: 1 if x['toss_decision'] == 'field' and x['winner'] != x['toss_winner']
        else (1 if x['toss_decision'] == 'bat' and x['winner'] == x['toss_winner'] else 0),
        axis=1
    )
```

* For each match mark **1** if team batting first wins, else **0**.

  * If toss winner chose to *field* and lost ⇒ batting‑first side won.
  * If toss winner chose to *bat* and won ⇒ batting‑first side also won.

```python
    matches_df['chasing_win'] = 1 - matches_df['batting_first_win']
```

* `chasing_win` is complementary (1 if chasing side won).

#### Aggregate counts

```python
    match_counts = matches_df['venue'].value_counts()
```

* Total matches per venue.

```python
    win_type_counts = matches_df.groupby('venue')[['batting_first_win', 'chasing_win']].sum()
```

* Sum wins by strategy at each venue.

#### Build summary table

```python
    summary = pd.DataFrame({
        'Matches': match_counts,
        'Wins Batting First': win_type_counts['batting_first_win'],
        'Wins Chasing': win_type_counts['chasing_win'],
        'Avg 1st Inn Score': avg_1st_innings_score
    }).fillna(0).astype({'Matches': int, 'Wins Batting First': int, 'Wins Chasing': int})
```

* Combine matches, wins, and average score into one DataFrame.
* Fill missing with 0 and ensure integer dtype.

```python
    summary['Bat First Win %'] = (summary['Wins Batting First'] / summary['Matches']) * 100
    summary['Chase Win %'] = (summary['Wins Chasing'] / summary['Matches']) * 100
```

* Compute win percentages for each strategy.

```python
    return summary.reset_index().rename(columns={'index': 'Venue'}).sort_values('Matches', ascending=False)
```

* Reset index to turn venue into a column, rename, sort by matches, and return.

---

### 2️⃣ Function `plot_venue_insight()`

*Displays the summary table and renders four visualisations.*

1. **Get data** via `corrected_venue_insights()` and display as Markdown.
2. **Ensure numeric columns** so Seaborn & matplotlib work reliably.
3. **Prepare slices**:

   * `df_bar` → top 10 venues by matches.
   * `df_score` → top 15 venues by average score.
   * `df_pct` → same as `df_bar` for percentage plot.
4. **Plot‑1 ( stacked bar )** – Total wins split into batting‑first vs chasing.
5. **Plot‑2 (horizontal bar)** – Average 1st‑innings scores.
6. **Plot‑3 (scatter)** – Relationship between matches played and average score.
7. **Plot‑4 (grouped bar)** – Bat First % vs Chase % per venue.

Each plot is wrapped with titles, axis labels, colour palettes and `tight_layout()` to avoid clipping.

---

### Example

```python
plot_venue_insight()  # renders table + 4 visuals
```

> 📌 Ensure global variables `matches_dataset` & `delivieres_dataset` are loaded before calling.

```python


def umpire_team_win_loss_summary(matches_df):
    # WIN SECTION
    umpire1_wins = matches_df.groupby(['umpire1', 'winner']).size().reset_index(name='wins')
    umpire2_wins = matches_df.groupby(['umpire2', 'winner']).size().reset_index(name='wins')
    all_umpire_wins = pd.concat([umpire1_wins, umpire2_wins])
    combined_wins = all_umpire_wins.groupby(['umpire1', 'winner'])['wins'].sum().reset_index()
    max_wins_per_umpire = combined_wins.loc[combined_wins.groupby('umpire1')['wins'].idxmax()]
    max_wins_per_umpire.columns = ['Umpire', 'Team with Max Wins', 'Wins']

    display(max_wins_per_umpire)

    # LOSS SECTION
    return max_wins_per_umpire

# Usage:
# display(umpire_team_win_loss_summary(matches_dataset))

```
##

---

## 🏟️ Venue Insight – Line‑by‑Line Guide

### 1️⃣ Function `corrected_venue_insights()`

```python
def corrected_venue_insights(matches_df, deliveries_df):
```

*Computes match counts, win splits, and average first‑innings scores for every IPL venue.*

```python
    merged_df = deliveries_df.merge(matches_df[['id', 'venue']], left_on='match_id', right_on='id', how='left')
```

* Join deliveries with venue names (from `matches_df`) on `match_id` → new DataFrame `merged_df`.

```python
    first_innings = merged_df[merged_df['inning'] == 1]
```

* Keep only first‑innings deliveries.

```python
    first_innings_total = first_innings.groupby(['match_id', 'venue'])['total_runs'].sum().reset_index()
```

* Sum runs for each match’s first innings.

```python
    avg_1st_innings_score = first_innings_total.groupby('venue')['total_runs'].mean()
```

* Average first‑innings total per venue.

#### Flag winner by strategy

```python
    matches_df['batting_first_win'] = matches_df.apply(
        lambda x: 1 if x['toss_decision'] == 'field' and x['winner'] != x['toss_winner']
        else (1 if x['toss_decision'] == 'bat' and x['winner'] == x['toss_winner'] else 0),
        axis=1
    )
```

* For each match mark **1** if team batting first wins, else **0**.

  * If toss winner chose to *field* and lost ⇒ batting‑first side won.
  * If toss winner chose to *bat* and won ⇒ batting‑first side also won.

```python
    matches_df['chasing_win'] = 1 - matches_df['batting_first_win']
```

* `chasing_win` is complementary (1 if chasing side won).

#### Aggregate counts

```python
    match_counts = matches_df['venue'].value_counts()
```

* Total matches per venue.

```python
    win_type_counts = matches_df.groupby('venue')[['batting_first_win', 'chasing_win']].sum()
```

* Sum wins by strategy at each venue.

#### Build summary table

```python
    summary = pd.DataFrame({
        'Matches': match_counts,
        'Wins Batting First': win_type_counts['batting_first_win'],
        'Wins Chasing': win_type_counts['chasing_win'],
        'Avg 1st Inn Score': avg_1st_innings_score
    }).fillna(0).astype({'Matches': int, 'Wins Batting First': int, 'Wins Chasing': int})
```

* Combine matches, wins, and average score into one DataFrame.
* Fill missing with 0 and ensure integer dtype.

```python
    summary['Bat First Win %'] = (summary['Wins Batting First'] / summary['Matches']) * 100
    summary['Chase Win %'] = (summary['Wins Chasing'] / summary['Matches']) * 100
```

* Compute win percentages for each strategy.

```python
    return summary.reset_index().rename(columns={'index': 'Venue'}).sort_values('Matches', ascending=False)
```

* Reset index to turn venue into a column, rename, sort by matches, and return.

---

### 2️⃣ Function `plot_venue_insight()`

*Displays the summary table and renders four visualisations.*

1. **Get data** via `corrected_venue_insights()` and display as Markdown.
2. **Ensure numeric columns** so Seaborn & matplotlib work reliably.
3. **Prepare slices**:

   * `df_bar` → top 10 venues by matches.
   * `df_score` → top 15 venues by average score.
   * `df_pct` → same as `df_bar` for percentage plot.
4. **Plot‑1 ( stacked bar )** – Total wins split into batting‑first vs chasing.
5. **Plot‑2 (horizontal bar)** – Average 1st‑innings scores.
6. **Plot‑3 (scatter)** – Relationship between matches played and average score.
7. **Plot‑4 (grouped bar)** – Bat First % vs Chase % per venue.

Each plot is wrapped with titles, axis labels, colour palettes and `tight_layout()` to avoid clipping.

---

### Example

```python
plot_venue_insight()  # renders table + 4 visuals
```

> 📌 Ensure global variables `matches_dataset` & `delivieres_dataset` are loaded before calling.

---

## 🧑‍⚖️ Umpire–Team Win Summary – Line‑by‑Line Guide

### Function `umpire_team_win_loss_summary()`

```python
def umpire_team_win_loss_summary(matches_df):
```

*Determines, for every on‑field umpire, which team has recorded the **most wins** under their supervision.*

```python
    umpire1_wins = matches_df.groupby(['umpire1', 'winner']).size().reset_index(name='wins')
```

* Count matches per (`umpire1`, `winner`) pair → DataFrame with columns: `umpire1`, `winner`, `wins`.

```python
    umpire2_wins = matches_df.groupby(['umpire2', 'winner']).size().reset_index(name='wins')
```

* Same count but for the second on‑field umpire.

```python
    all_umpire_wins = pd.concat([umpire1_wins, umpire2_wins])
```

* Stack both tables so each row now says: *“Umpire X – Team Y – Wins N”* (column name for umpire differs in the two halves).

```python
    combined_wins = all_umpire_wins.groupby(['umpire1', 'winner'])['wins'].sum().reset_index()
```

* **Key normalisation**: After concat, the column is `umpire1` in both halves (because the second frame had `umpire2` → now mismatched). Here we group by that umpire column and sum wins to merge duplicates.

```python
    max_wins_per_umpire = combined_wins.loc[combined_wins.groupby('umpire1')['wins'].idxmax()]
```

* For each umpire pick the row with the **maximum wins** (i.e.
  the team that won most under them).

```python
    max_wins_per_umpire.columns = ['Umpire', 'Team with Max Wins', 'Wins']
```

* Rename columns for readability.

```python
    display(max_wins_per_umpire)
```

* Show the result inline if running inside a notebook.

```python
    # LOSS SECTION
    return max_wins_per_umpire
```

* (Placeholder comment for a future “loss” calculation.)
* Function returns the table so it can be used programmatically.

### Usage Example

```python
df_max_wins = umpire_team_win_loss_summary(matches_dataset)
display(df_max_wins)
```

> 📌 **Tip:** To extend this function for *losses*, copy the same logic but count where `winner != team` once you derive the opponents.
