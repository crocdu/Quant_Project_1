# Jing
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_signals_with_trigger_60min(data, threshold, initial_position, alpha=2):
    """
    Generates trading signals based on volatility changes.

    Parameters:
        data (pd.DataFrame): Input data containing historical volatility and futures data.
        threshold (float): Threshold value to trigger a position change.
        initial_position (int): Initial trading position.
        alpha (int, optional): Multiplier for signal sensitivity. Default is 2.

    Returns:
        pd.DataFrame: Data with generated trading signals and positions.
    """
    data['changed_score'] = data['HV10'] - data['HV30']
    results = []

    # Convert datetime column to proper format
    data['datetime'] = pd.to_datetime(data['dt'])
    data = data.sort_values(by=['datetime'])
    unique_dates = data['date'].dt.date.unique()

    # Define trade times for different frequencies
    trade_times = [
        pd.Timestamp("21:01:00").time(), pd.Timestamp("22:01:00").time(),
        pd.Timestamp("23:01:00").time(), pd.Timestamp("00:01:00").time(),
        pd.Timestamp("01:01:00").time(), pd.Timestamp("02:01:00").time(),
        pd.Timestamp("09:01:00").time(), pd.Timestamp("10:01:00").time(),
        pd.Timestamp("11:16:00").time(), pd.Timestamp("14:16:00").time(),
        pd.Timestamp("14:59:00").time()
    ]

    signals = []
    positions = []
    position = initial_position
    previous_score = 0

    for unique_date in unique_dates:
        daily_data = data[data['date'].dt.date == unique_date].copy()
        daily_data = daily_data.set_index('datetime')
        daily_data = daily_data[np.isin(daily_data.index.time, trade_times)]
        daily_data = daily_data.groupby(daily_data.index).agg({
            'changed_score': 'last',
            'ATM Volatility': 'last',
            'Futures': 'last',
            'contract': 'last',
            'HV5': 'last',
            'HV10': 'last',
            'HV15': 'last',
            'HV20': 'last',
            'HV25': 'last',
            'HV30': 'last',
            'group': 'last',
        }).dropna().reset_index()

        daily_signals = []
        daily_positions = []

        for _, row in daily_data.iterrows():
            current_score = row['changed_score']
            cumulative_diff = current_score - previous_score

            if row['group'] == 0:
                position = 0
                previous_score = 0
                daily_signals.append(0)
                daily_positions.append(0)
                continue

            if len(positions) == 0 or position == 0:
                if cumulative_diff >= (alpha + 1) * threshold:
                    signal = -math.floor((abs(cumulative_diff) - threshold * alpha) / threshold)
                    position -= abs(signal)
                elif cumulative_diff <= -(alpha + 1) * threshold:
                    signal = math.floor((abs(cumulative_diff) - threshold * alpha) / threshold)
                    position += abs(signal)
                else:
                    signal = 0
            elif position < 0:
                if cumulative_diff <= -threshold:
                    if current_score < -((alpha + 1) * threshold):
                        signal = math.floor((abs(current_score) - threshold * alpha) / threshold) + abs(positions[-1])
                    else:
                        signal = min(math.floor(abs(cumulative_diff) / threshold), -positions[-1])
                    position += abs(signal)
                elif cumulative_diff >= threshold:
                    signal = -math.floor(abs(cumulative_diff) / threshold)
                    position -= abs(signal)
                else:
                    signal = 0
            elif position > 0:
                if cumulative_diff >= threshold:
                    if current_score >= (alpha + 1) * threshold:
                        signal = -(math.floor((abs(current_score) - threshold * alpha) / threshold) + abs(
                            positions[-1]))
                    else:
                        signal = -min(math.floor(abs(cumulative_diff) / threshold), positions[-1])
                    position -= abs(signal)
                elif cumulative_diff <= -threshold:
                    signal = math.floor(abs(cumulative_diff) / threshold)
                    position += abs(signal)
                else:
                    signal = 0

            previous_score = -((abs(position) + alpha) * threshold) * np.sign(position)
            daily_signals.append(signal)
            daily_positions.append(position)

        signals.extend(daily_signals)
        positions.extend(daily_positions)
        daily_data['signal'] = daily_signals
        daily_data['position'] = daily_positions
        results.append(daily_data)

    return pd.concat(results).reset_index(drop=True)


def calculate_strategy_returns(data):
    """
    Calculate strategy returns.

    Parameters:
        data (pd.DataFrame): Data containing trading signals and positions.

    Returns:
        pd.DataFrame: Data with calculated strategy returns.
    """
    data['strategy_returns'] = data['position'].shift(1) * data['changed_score']
    return data


def calculate_cumulative_returns(data):
    """
    Calculate cumulative returns from strategy returns.

    Parameters:
        data (pd.DataFrame): Data containing strategy returns.

    Returns:
        pd.DataFrame: Data with calculated cumulative returns.
    """
    if 'strategy_returns' not in data.columns:
        raise KeyError("The column 'strategy_returns' is missing!")

    data['strategy_returns'] = data['strategy_returns'].fillna(0)
    data['cumulative_returns'] = data['strategy_returns'].cumsum()
    return data


def main():
    file_path = r'C:/Users/jing.yu/PycharmProjects/pythonProject/Project_1/Strategy_3/output/ag_final_mode_3.csv'
    data = pd.read_csv(file_path)

    required_columns = ['date', 'contract', 'HV5', 'HV10', 'HV15', 'HV20', 'HV25', 'HV30', 'ATM Volatility', 'Futures']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")

    data['date'] = pd.to_datetime(data['date'])
    threshold = 0.005
    initial_position = 0

    data = generate_signals_with_trigger_60min(data, threshold, initial_position)
    data = calculate_strategy_returns(data)
    data = calculate_cumulative_returns(data)

    output_file_path = r'C:/Users/jing.yu/PycharmProjects/pythonProject/Project_1/Strategy_3/output/result/ag/ag_output.csv'
    data.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}!")


if __name__ == "__main__":
    main()






import matplotlib.pyplot as plt

# Load processed data
input_file = r'C:/Users/jing.yu/PycharmProjects/pythonProject/Project_1/Strategy_3/output/result/ag/ag_output.csv'
data = pd.read_csv(input_file)

# Convert datetime column to proper format
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data = data.sort_values(by='datetime')

# Ensure required columns exist and drop missing values
required_columns = ['cumulative_returns', 'HV10', 'HV30', 'ATM Volatility', 'Futures', 'signal', 'position']
data = data.dropna(subset=required_columns)

# Set plotting style
plt.style.use('ggplot')

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 14), gridspec_kw={'height_ratios': [2, 2, 1, 1]}, sharex=True)

# Plot cumulative returns
axes[0].plot(data['datetime'], data['cumulative_returns'], color='blue', linewidth=1, label="Cumulative Returns")
axes[0].set_title("HV10 - HV30 Cumulative Returns")
axes[0].set_ylabel("Returns")
axes[0].legend()

# Plot Futures prices
axes[1].plot(data['datetime'], data['Futures'], color='blue', linewidth=1, label="Futures Price")
axes[1].set_ylabel("Futures Price")
axes[1].legend()

# Plot signal and position with dual y-axes
axes1_right = axes[2].twinx()
axes[2].plot(data['datetime'], data['changed_score'], color='blue', linewidth=1, label="Signal")
axes1_right.step(data['datetime'], -data['position'], color='black', linestyle=':', linewidth=1, where='post', label="- Position")

axes[2].set_ylabel("Signal")
axes1_right.set_ylabel("Position", color='black')
axes1_right.tick_params(axis='y', labelcolor='black')
axes[2].legend(loc="upper left")
axes1_right.legend(loc="upper right")

# Plot ATM Volatility
axes[3].plot(data['datetime'], data['ATM Volatility'], color='blue', linewidth=1, label="ATM Volatility")
axes[3].set_ylabel("ATM Volatility")
axes[3].legend()

# Adjust layout and show the plot
plt.subplots_adjust(top=0.97, hspace=0.1)
plt.show()






import numpy as np

# Load strategy results
input_file = r"C:/Users/jing.yu/PycharmProjects/pythonProject/Project_1/Strategy_3/output/result/ag/ag_output.csv"
data = pd.read_csv(input_file)

# Ensure strategy_returns exists
if 'strategy_returns' not in data.columns:
    raise KeyError("Missing column: 'strategy_returns' in dataset")

# Compute key metrics
pnl = data['strategy_returns'].sum()  # Profit and Loss
pot = (pnl / data['signal'].abs().sum()) * 100  # Profit over Turnover (%)
num_trades = data['signal'].abs().sum()  # Number of trades

# Compute maximum drawdown
cumulative_max = data['cumulative_returns'].cummax()
max_drawdown = np.nanmin(data['cumulative_returns'] - cumulative_max)

# Store results in DataFrame
overall_summary = pd.DataFrame({
    "Metric": ["PnL", "PoT (%)", "Threshold", "Alpha", "Number of Trades", "Max Drawdown"],
    "Value": [round(pnl, 4), round(pot, 4), 0.005, 2, int(num_trades), round(max_drawdown, 4)]
})

# Display overall metrics
print("\nOverall Strategy Metrics:")
print(overall_summary)

# Save output
output_file = r"C:/Users/jing.yu/PycharmProjects/pythonProject/Project_1/Strategy_3/output/result/ag/HV10-HV30.csv"
overall_summary.to_csv(output_file, index=False)
print(f"Metrics saved to {output_file}")
