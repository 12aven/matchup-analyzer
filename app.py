import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup, team_game_logs, statcast
import statsapi
import json
import traceback
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Marlins Matchup Analyzer",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 24px; color: #FF6B6B; font-weight: bold;}
    .sub-header {font-size: 20px; color: #4ECDC4; margin-top: 20px;}
    .stat-box {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

def get_team_roster(team_id=146):
    """Get current Marlins roster"""
    try:
        # Get roster data as a list of player dictionaries
        roster_data = statsapi.get('team_roster', {'teamId': team_id, 'rosterType': 'active'})
        roster_list = []
        
        for player in roster_data['roster']:
            player_info = {
                'id': player['person']['id'],
                'fullName': player['person']['fullName'],
                'primaryNumber': player.get('jerseyNumber', ''),
                'position': player.get('position', {}).get('name', 'Pitcher' if player.get('position', {}).get('code') == '1' else 'Position Player')
            }
            roster_list.append(player_info)
            
        return pd.DataFrame(roster_list)
    except Exception as e:
        st.error(f"Error loading roster: {e}")
        return pd.DataFrame()

def get_team_schedule(team_id=146):
    """Get team schedule"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        # Set end date to 30 days from today
        end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Get schedule data using the correct endpoint
        params = {
            'sportId': 1,  # 1 is for MLB
            'teamId': team_id,
            'startDate': today,
            'endDate': end_date,
            'hydrate': 'team,linescore,decisions,probablePitcher',
            'fields': 'dates,date,games,gamePk,teams,away,home,team,name,shortName,gameDate,status,abstractGameState,detailedState,isTie,gameNumber,doubleHeader,dayNight,description,inning,inningState,reason,abstractGameCode'
        }
        schedule = statsapi.get('schedule', params)
        
        # Extract games from the schedule and format dates
        games = []
        if 'dates' in schedule:
            for date in schedule['dates']:
                for game in date.get('games', []):
                    # Format the game date to be more readable
                    if 'gameDate' in game:
                        try:
                            # Parse the ISO format date and reformat it
                            dt = datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00'))
                            game['gameDate'] = dt.strftime('%Y-%m-%d %H:%M')
                        except (ValueError, TypeError):
                            pass
                    games.append(game)
        
        return pd.DataFrame(games)
    except Exception as e:
        st.error(f"Error loading schedule: {str(e)}")
        return pd.DataFrame()

def get_batter_stats(player_id, days=30):
    """Get batter stats for last N days using statcast data"""
    try:
        player_id = int(player_id)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        print(f"Fetching batter stats for player {player_id} from {start_date} to {end_date}")
        
        try:
            df = statcast_batter(start_date, end_date, player_id)
            
            # Debug: Print available columns
            if df is not None and not df.empty:
                print(f"Available columns in batter statcast data: {df.columns.tolist()}")
                print(f"Sample events: {df['events'].dropna().unique() if 'events' in df.columns else 'No events column'}")
            else:
                print("No data returned from statcast_batter")
                return pd.DataFrame()
            
            # Ensure we have the minimum required columns
            required_cols = ['events', 'game_date', 'game_pk', 'at_bat_number']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Create a safe copy of the dataframe
            df = df.copy()
            
            # Initialize all event types to 0
            df['is_hit'] = 0
            df['is_strikeout'] = 0
            df['is_walk'] = 0
            df['is_home_run'] = 0
            
            # Safely set values based on events
            df.loc[df['events'].isin(['single', 'double', 'triple', 'home_run']), 'is_hit'] = 1
            df.loc[df['events'] == 'strikeout', 'is_strikeout'] = 1
            df.loc[df['events'].isin(['walk', 'intent_walk', 'hit_by_pitch']), 'is_walk'] = 1
            df.loc[df['events'] == 'home_run', 'is_home_run'] = 1
            
            # Group by game
            group_cols = {
                'at_bat_number': 'nunique',  # ABs
                'is_hit': 'sum',             # Hits
                'is_strikeout': 'sum',       # Strikeouts
                'is_walk': 'sum',            # Walks
                'is_home_run': 'sum',        # HRs
                'game_date': 'first'         # Game date
            }
            
            # Add optional columns if they exist
            optional_cols = {
                'home_team': 'first',
                'away_team': 'first',
                'home_score': 'first',
                'away_score': 'first',
                'launch_speed': 'mean',
                'launch_angle': 'mean',
                'hit_distance_sc': 'mean'
            }
            
            # Only include columns that exist in the dataframe
            for col, agg in optional_cols.items():
                if col in df.columns:
                    group_cols[col] = agg
            
            game_stats = df.groupby('game_pk').agg(group_cols).reset_index()
            
            # Rename columns to match expected format
            rename_cols = {
                'at_bat_number': 'atBats',
                'is_hit': 'hits',
                'is_strikeout': 'strikeOuts',
                'is_walk': 'baseOnBalls',
                'is_home_run': 'homeRuns',
                'game_date': 'gameDate',
                'home_team': 'homeTeam',
                'away_team': 'awayTeam',
                'home_score': 'homeScore',
                'away_score': 'awayScore',
                'launch_speed': 'avgExitVelocity',
                'launch_angle': 'avgLaunchAngle',
                'hit_distance_sc': 'avgHitDistance'
            }
            
            # Only include columns that exist in the dataframe
            rename_cols = {k: v for k, v in rename_cols.items() if k in game_stats.columns}
            game_stats = game_stats.rename(columns=rename_cols)
            
            return game_stats
            
        except Exception as api_error:
            print(f"API Error in get_batter_stats: {str(api_error)}")
            traceback.print_exc()
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in get_batter_stats: {str(e)}")
        return pd.DataFrame()

def get_pitcher_stats(player_id, days=30):
    """Get pitcher stats for last N days using statcast data"""
    try:
        player_id = int(player_id)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        print(f"Fetching pitcher stats for player {player_id} from {start_date} to {end_date}")
        
        try:
            df = statcast_pitcher(start_date, end_date, player_id)
            
            # Debug: Print available columns
            if df is not None and not df.empty:
                print(f"Available columns in pitcher statcast data: {df.columns.tolist()}")
                print(f"Sample events: {df['events'].dropna().unique() if 'events' in df.columns else 'No events column'}")
            else:
                print("No data returned from statcast_pitcher")
                return pd.DataFrame()
            
            # Ensure we have the minimum required columns
            required_cols = ['events', 'at_bat_number', 'pitch_number', 'game_pk']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Create a safe copy of the dataframe
            df = df.copy()
            
            # Initialize all event types to 0
            df['is_hit'] = 0
            df['is_strikeout'] = 0
            df['is_walk'] = 0
            df['is_home_run'] = 0
            
            # Safely set values based on events
            df.loc[df['events'].isin(['single', 'double', 'triple', 'home_run']), 'is_hit'] = 1
            df.loc[df['events'] == 'strikeout', 'is_strikeout'] = 1
            df.loc[df['events'].isin(['walk', 'intent_walk', 'hit_by_pitch']), 'is_walk'] = 1
            df.loc[df['events'] == 'home_run', 'is_home_run'] = 1
            
            # Group by game
            group_cols = {
                'at_bat_number': 'nunique',  # Batters faced
                'pitch_number': 'count',     # Total pitches
                'is_strikeout': 'sum',       # Strikeouts
                'is_walk': 'sum',            # Walks
                'is_home_run': 'sum',        # HRs
                'is_hit': 'sum'              # Hits
            }
            
            # Add optional columns if they exist
            optional_cols = {
                'game_date': 'first',
                'home_team': 'first',
                'away_team': 'first',
                'home_score': 'first',
                'away_score': 'first',
                'release_speed': 'mean',
                'release_spin_rate': 'mean',
                'effective_speed': 'mean'
            }
            
            # Only include columns that exist in the dataframe
            for col, agg in optional_cols.items():
                if col in df.columns:
                    group_cols[col] = agg
            
            game_stats = df.groupby('game_pk').agg(group_cols).reset_index()
            
            # Calculate innings pitched (estimate: 3 outs per inning, 3 batters per out)
            game_stats['inningsPitched'] = game_stats['at_bat_number'] / 3
            
            # Rename columns to match expected format
            rename_cols = {
                'at_bat_number': 'battersFaced',
                'pitch_number': 'totalPitches',
                'is_strikeout': 'strikeOuts',
                'is_walk': 'baseOnBalls',
                'is_home_run': 'homeRuns',
                'is_hit': 'hits',
                'game_date': 'gameDate',
                'home_team': 'homeTeam',
                'away_team': 'awayTeam',
                'home_score': 'homeScore',
                'away_score': 'awayScore',
                'release_speed': 'avgFastballVelocity',
                'release_spin_rate': 'avgSpinRate',
                'effective_speed': 'avgEffectiveSpeed'
            }
            
            # Only include columns that exist in the dataframe
            rename_cols = {k: v for k, v in rename_cols.items() if k in game_stats.columns}
            game_stats = game_stats.rename(columns=rename_cols)
            
            return game_stats
            
        except Exception as api_error:
            print(f"API Error in get_pitcher_stats: {str(api_error)}")
            traceback.print_exc()
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in get_pitcher_stats: {str(e)}")
        return pd.DataFrame()

def calculate_advanced_stats(df):
    """Calculate advanced batting statistics"""
    if df.empty or not isinstance(df, pd.DataFrame):
        return {}
        
    stats = {}
    
    # Initialize stats with defaults
    stats['PA'] = len(df)
    stats['AB'] = len(df.get('atBats', df))  # Use atBats if available, otherwise use all rows
    stats['H'] = df.get('hits', 0)
    stats['1B'] = len(df[df['events'] == 'single']) if 'events' in df.columns else 0
    stats['2B'] = len(df[df['events'] == 'double']) if 'events' in df.columns else 0
    stats['3B'] = len(df[df['events'] == 'triple']) if 'events' in df.columns else 0
    stats['HR'] = len(df[df['events'] == 'home_run']) if 'events' in df.columns else 0
    stats['BB'] = len(df[df['events'].isin(['walk', 'intent_walk', 'hit_by_pitch'])]) if 'events' in df.columns else 0
    stats['SO'] = len(df[df['events'] == 'strikeout']) if 'events' in df.columns else 0
    
    # Calculate rates
    if stats['AB'] > 0:
        stats['AVG'] = stats['H'] / stats['AB']
        stats['SLG'] = (stats['1B'] + 2*stats['2B'] + 3*stats['3B'] + 4*stats['HR']) / stats['AB']
    else:
        stats['AVG'] = 0
        stats['SLG'] = 0
    
    # Calculate K% and BB%
    if stats['PA'] > 0:
        stats['K%'] = (stats['SO'] / stats['PA']) * 100
        stats['BB%'] = (stats['BB'] / stats['PA']) * 100
    else:
        stats['K%'] = 0
        stats['BB%'] = 0
    
    # Calculate wOBA (simplified version)
    if stats['PA'] > 0:
        stats['wOBA'] = (
            0.7 * stats['BB'] + 
            0.9 * stats['1B'] + 
            1.25 * stats['2B'] + 
            1.6 * stats['3B'] + 
            2.0 * stats['HR']
        ) / stats['PA']
    else:
        stats['wOBA'] = 0
    
    # Format numbers for display
    for key in ['AVG', 'SLG', 'wOBA']:
        if key in stats:
            stats[key] = round(stats[key], 3)
    for key in ['K%', 'BB%']:
        if key in stats:
            stats[key] = round(stats[key], 1)
    
    return stats

def get_platoon_splits(player_id: int, is_batter: bool = True, days: int = 365) -> pd.DataFrame:
    """Get platoon splits for a batter or pitcher"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    if is_batter:
        df = statcast_batter(start_date, end_date, player_id)
        split_col = 'p_throws'  # Batter's splits by pitcher handedness
    else:
        df = statcast_pitcher(start_date, end_date, player_id)
        split_col = 'stand'  # Pitcher's splits by batter handedness
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate metrics for each split
    splits = []
    for hand in ['R', 'L']:
        split_df = df[df[split_col] == hand].copy()
        
        # Calculate wOBA (simplified version)
        pa = len(split_df)
        if pa == 0:
            continue
            
        # Calculate basic stats
        stats = {
            'hand': hand,
            'PA': pa,
            'AB': len(split_df[~split_df['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt'])]),
            'H': len(split_df[split_df['events'].isin(['single', 'double', 'triple', 'home_run'])]),
            '1B': len(split_df[split_df['events'] == 'single']),
            '2B': len(split_df[split_df['events'] == 'double']),
            '3B': len(split_df[split_df['events'] == 'triple']),
            'HR': len(split_df[split_df['events'] == 'home_run']),
            'BB': len(split_df[split_df['events'].isin(['walk', 'intent_walk'])]),
            'HBP': len(split_df[split_df['events'] == 'hit_by_pitch']),
            'SO': len(split_df[split_df['events'] == 'strikeout'])
        }
        
        # Calculate wOBA (simplified)
        if stats['AB'] > 0:
            stats['AVG'] = stats['H'] / stats['AB']
            stats['SLG'] = (stats['1B'] + 2*stats['2B'] + 3*stats['3B'] + 4*stats['HR']) / stats['AB']
            stats['wOBA'] = (0.7*stats['BB'] + 0.9*stats['1B'] + 1.25*stats['2B'] + 1.6*stats['3B'] + 2.0*stats['HR']) / stats['PA']
            stats['ISO'] = (stats['2B'] + 2*stats['3B'] + 3*stats['HR']) / stats['AB']
        
        splits.append(stats)
    
    return pd.DataFrame(splits)

def get_batted_ball_profile(player_id: int, is_batter: bool = True, days: int = 365) -> Dict:
    """Get batted ball profile (GB%, LD%, FB%)"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    if is_batter:
        df = statcast_batter(start_date, end_date, player_id)
    else:
        df = statcast_pitcher(start_date, end_date, player_id)
    
    if df.empty or 'bb_type' not in df.columns:
        return {}
    
    # Filter for batted balls
    batted_balls = df[df['bb_type'].notna()].copy()
    total_bb = len(batted_balls)
    
    if total_bb == 0:
        return {}
    
    # Calculate batted ball rates
    bb_types = batted_balls['bb_type'].value_counts()
    bb_rates = (bb_types / total_bb * 100).round(1).to_dict()
    
    # Calculate average metrics
    avg_metrics = {
        'exit_velocity': batted_balls['launch_speed'].mean(),
        'launch_angle': batted_balls['launch_angle'].mean(),
        'distance': batted_balls['hit_distance_sc'].mean()
    }
    
    return {
        'rates': bb_rates,
        'avg_metrics': avg_metrics,
        'batted_balls': batted_balls[['game_date', 'bb_type', 'launch_speed', 'launch_angle', 'hit_distance_sc', 'events']]
    }

def main():
    st.title("⚾ Marlins Matchup Analyzer")
    
    # Sidebar
    st.sidebar.header("Player Selection")
    
    # Date range selector
    days = st.sidebar.slider("Date Range (days)", 7, 365, 30, 7,
                           help="Select the number of days to look back for stats")
    
    # Get team schedule
    try:
        schedule = get_team_schedule()
        if not schedule.empty:
            game_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M').strftime('%m/%d/%Y') 
                         for date in schedule['gameDate'].unique()]
            selected_date = st.sidebar.selectbox("Select Game Date", game_dates, index=0)
    except Exception as e:
        st.sidebar.error(f"Error loading schedule: {e}")
        selected_date = datetime.now().strftime('%m/%d/%Y')
    
    # Get roster
    try:
        roster = get_team_roster()
        batters = roster[roster['position'] != 'Pitcher']
        pitchers = roster[roster['position'] == 'Pitcher']
        
        # Player selection
        selected_batter = st.sidebar.selectbox(
            "Select Batter",
            batters['fullName'].sort_values().tolist()
        )
        
        selected_pitcher = st.sidebar.selectbox(
            "Select Pitcher",
            pitchers['fullName'].sort_values().tolist()
        )
        
        # Get player IDs
        batter_id = batters[batters['fullName'] == selected_batter]['id'].iloc[0]
        pitcher_id = pitchers[pitchers['fullName'] == selected_pitcher]['id'].iloc[0]
        
    except Exception as e:
        st.sidebar.error(f"Error loading roster: {e}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Recent Performance", 
        "Platoon Splits", 
        "Batted Ball Profile"
    ])
    
    with tab1:
        # Recent Performance
        st.header(f"Recent Performance (Last {days} Days)")
        
        # Batter's recent performance
        st.markdown(f"### {selected_batter} - Batter")
        try:
            batter_stats = get_batter_stats(batter_id, days=days)
            if not batter_stats.empty:
                # Calculate metrics using the correct column names
                batter_stats['gameDate'] = pd.to_datetime(batter_stats['gameDate']).dt.strftime('%Y-%m-%d')
                
                # Create a line chart for batter's performance
                fig = px.line(batter_stats, x='gameDate', y='hits', 
                            title=f"{selected_batter} - Hits by Game")
                st.plotly_chart(fig)
                
                # Show detailed stats
                st.dataframe(batter_stats[['gameDate', 'atBats', 'hits', 'homeRuns', 'strikeOuts', 'baseOnBalls']]
                            .rename(columns={
                                'gameDate': 'Game Date',
                                'atBats': 'AB',
                                'hits': 'H',
                                'homeRuns': 'HR',
                                'strikeOuts': 'SO',
                                'baseOnBalls': 'BB'
                            }))
            else:
                st.warning("No recent batting data available.")
        except Exception as e:
            st.error(f"Error loading batter stats: {str(e)}")
        
        # Pitcher's recent performance
        st.markdown(f"### {selected_pitcher} - Pitcher")
        try:
            pitcher_stats = get_pitcher_stats(pitcher_id, days=days)
            if not pitcher_stats.empty:
                # Calculate metrics using the correct column names
                pitcher_stats['gameDate'] = pd.to_datetime(pitcher_stats['gameDate']).dt.strftime('%Y-%m-%d')
                
                # Create a line chart for pitcher's performance
                fig = px.line(pitcher_stats, x='gameDate', y='strikeOuts', 
                            title=f"{selected_pitcher} - Strikeouts by Game")
                st.plotly_chart(fig)
                
                # Show detailed stats
                st.dataframe(pitcher_stats[['gameDate', 'battersFaced', 'hits', 'homeRuns', 'strikeOuts', 'baseOnBalls', 'totalPitches', 'inningsPitched']]
                            .rename(columns={
                                'gameDate': 'Game Date',
                                'battersFaced': 'BF',
                                'hits': 'H',
                                'homeRuns': 'HR',
                                'strikeOuts': 'SO',
                                'baseOnBalls': 'BB',
                                'totalPitches': 'Pitches',
                                'inningsPitched': 'IP'
                            }))
            else:
                st.warning("No recent pitching data available.")
        except Exception as e:
            st.error(f"Error loading pitcher stats: {str(e)}")
    
    with tab2:
        # Platoon Splits
        st.header("Platoon Splits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_batter} vs. Pitcher Handedness")
            batter_splits = get_platoon_splits(batter_id, is_batter=True, days=days)
            if not batter_splits.empty:
                st.dataframe(batter_splits)
            else:
                st.warning("No platoon split data available for batter.")
        
        with col2:
            st.subheader(f"{selected_pitcher} vs. Batter Handedness")
            pitcher_splits = get_platoon_splits(pitcher_id, is_batter=False, days=days)
            if not pitcher_splits.empty:
                st.dataframe(pitcher_splits)
            else:
                st.warning("No platoon split data available for pitcher.")
    
    with tab3:
        # Batted Ball Profile
        st.header("Batted Ball Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_batter}'s Batted Ball Profile")
            batter_profile = get_batted_ball_profile(batter_id, is_batter=True, days=days)
            if batter_profile and 'rates' in batter_profile:
                # Create donut chart
                fig = px.pie(
                    names=list(batter_profile['rates'].keys()),
                    values=list(batter_profile['rates'].values()),
                    hole=0.4,
                    title="Batted Ball Distribution"
                )
                st.plotly_chart(fig)
                
                # Show average metrics
                st.metric("Avg Exit Velo", f"{batter_profile['avg_metrics'].get('exit_velocity', 0):.1f} mph")
                st.metric("Avg Launch Angle", f"{batter_profile['avg_metrics'].get('launch_angle', 0):.1f}°")
                st.metric("Avg Distance", f"{batter_profile['avg_metrics'].get('distance', 0):.1f} ft")
            else:
                st.warning("No batted ball data available for batter.")
        
        with col2:
            st.subheader(f"{selected_pitcher}'s Batted Ball Profile")
            pitcher_profile = get_batted_ball_profile(pitcher_id, is_batter=False, days=days)
            if pitcher_profile and 'rates' in pitcher_profile:
                # Create donut chart
                fig = px.pie(
                    names=list(pitcher_profile['rates'].keys()),
                    values=list(pitcher_profile['rates'].values()),
                    hole=0.4,
                    title="Batted Ball Distribution"
                )
                st.plotly_chart(fig)
                
                # Show average metrics
                st.metric("Avg Exit Velo", f"{pitcher_profile['avg_metrics'].get('exit_velocity', 0):.1f} mph")
                st.metric("Avg Launch Angle", f"{pitcher_profile['avg_metrics'].get('launch_angle', 0):.1f}°")
                st.metric("Avg Distance", f"{pitcher_profile['avg_metrics'].get('distance', 0):.1f} ft")
            else:
                st.warning("No batted ball data available for pitcher.")

if __name__ == "__main__":
    main()
