import pandas as pd 
import numpy as np

players_df = pd.read_csv("D:/Sexy laptop Data/FIFA2026/data/Transfermkt/players.csv")




def create_squad_strength_score(players_df):

    """
    Clean Transfermarkt player data
    Create national team level features
    """

    df = players_df.copy()


    # -----------------------------------
    # 1. Keep latest player record
    # -----------------------------------

    df = (
        df.sort_values("last_season")
        .drop_duplicates(
            subset="player_id",
            keep="last"
        )
    )


    # -----------------------------------
    # 2. Select required columns
    # -----------------------------------

    cols = [
        "player_id",
        "current_national_team_id",
        "country_of_citizenship",
        "market_value_in_eur",
        "date_of_birth",
        "international_caps",
        "position"
    ]

    df = df[cols]


    # -----------------------------------
    # 3. Clean market value
    # -----------------------------------

    df["market_value_in_eur"] = pd.to_numeric(
        df["market_value_in_eur"],
        errors="coerce"
    )

    df["market_value_in_eur"] = (
        df["market_value_in_eur"]
        .fillna(0)
    )


    # -----------------------------------
    # 4. Remove players without national team
    # -----------------------------------

    df = df.dropna(
        subset=[
            "current_national_team_id"
        ]
    )


    # -----------------------------------
    # 5. Calculate age
    # -----------------------------------

    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"],
        errors="coerce"
    )


    today = pd.Timestamp.today()

    df["age"] = (
        (today - df["date_of_birth"])
        .dt.days / 365
    )


    # -----------------------------------
    # 6. Aggregate by national team
    # -----------------------------------

    squad = (

        df.groupby(
            [
                "current_national_team_id",
            ]
        )
        .agg(

            squad_size = (
                "player_id",
                "count"
            ),

            squad_market_value = (
                "market_value_in_eur",
                "sum"
            ),

            average_player_value = (
                "market_value_in_eur",
                "mean"
            ),

            average_age = (
                "age",
                "mean"
            ),

            average_caps = (
                "international_caps",
                "mean"
            )

        )

        .reset_index()

    )


    # -----------------------------------
    # 7. Create squad score
    # -----------------------------------

    squad["value_score"] = (
        squad["squad_market_value"]
        /
        squad["squad_market_value"].max()
    )


    squad["depth_score"] = (
        squad["squad_size"]
        /
        squad["squad_size"].max()
    )


    squad["experience_score"] = (
        squad["average_caps"]
        /
        squad["average_caps"].max()
    )


    squad["squad_strength_score"] = (

        squad["value_score"] * 0.6

        +

        squad["depth_score"] * 0.2

        +

        squad["experience_score"] * 0.2

    )


    # -----------------------------------
    # 8. Sort
    # -----------------------------------

    squad = squad.sort_values(
        "squad_strength_score",
        ascending=False
    )


    print(
        squad.head(20)
    )


    return squad



squad_table = create_squad_strength_score(players_df)