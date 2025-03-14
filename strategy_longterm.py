# Jing
import numpy as np
import pandas as pd
import math


# Small Alpha Logic
def generate_signals_with_trigger_new(data, threshold, initial_position, alpha=2):
    """ Generate trading signals based on threshold logic. """
    data['changed_score'] = data['HV5'] - data['HV15']
    results = []

    for code, group in data.groupby('code'):
        group = group.copy().reset_index(drop=True)
        signals, positions = [], []
        position, trading_started = initial_position, False
        previous_score = 0
        signal = 0

        for index, row in group.iterrows():
            current_score = row['changed_score']
            cumulative_diff = current_score - previous_score

            if index >= len(group) - 1:
                signals.append(-positions[-1])
                positions.append(0)
                continue

            if len(positions) == 0 or positions[-1] == 0:
                if cumulative_diff >= (alpha + 1) * threshold:
                    signal = -math.floor((abs(cumulative_diff) - threshold * alpha) / threshold)
                    position -= abs(signal)
                elif cumulative_diff <= -(alpha + 1) * threshold:
                    signal = math.floor((abs(cumulative_diff) - threshold * alpha) / threshold)
                    position += abs(signal)
                else:
                    signal = 0
            elif positions[-1] > 0:
                if cumulative_diff >= threshold:
                    if current_score >= (alpha + 1) * threshold:
                        signal = -(math.floor((abs(current_score) - (threshold * alpha)) / threshold) + abs(
                            positions[-1]))
                    else:
                        signal = -min(math.floor(abs(cumulative_diff) / threshold), positions[-1])
                    position -= abs(signal)
                elif cumulative_diff <= -threshold:
                    signal = math.floor(abs(cumulative_diff) / threshold)
                    position += abs(signal)
                else:
                    signal = 0
            elif positions[-1] < 0:
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

            previous_score = -((abs(position) + alpha) * threshold) * np.sign(position)
            signals.append(signal)
            positions.append(position)

        group['signal'] = signals
        group['position'] = positions
        results.append(group)

    return pd.concat(results).reset_index(drop=True)


# Strategy return calculations
def calculate_strategy_returns(data):
    """ Calculate strategy returns based on ATM Volatility change. """
    if 'ATM_Vol_Change' not in data.columns:
        data['ATM_Vol_Change'] = data['ATM Volatility'].diff()
    data['Strategy_Returns'] = data['position'].shift(1) * data['ATM_Vol_Change']
    return data


# Cumulative returns calculation
def calculate_cumulative_returns(data):
    """ Calculate cumulative returns for the strategy. """
    if 'Strategy_Returns' not in data.columns:
        raise KeyError("The column 'Strategy_Returns' is missing!")

    data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
    results = []
    last_cumulative_return = 0

    for code, group in data.groupby('code'):
        group = group.copy()
        group['Cumulative_Returns'] = group['Strategy_Returns'].cumsum()
        group['Cumulative_Returns'] += last_cumulative_return
        last_cumulative_return = group['Cumulative_Returns'].iloc[-1]
        results.append(group)

    return pd.concat(results).sort_index()


# Data loading function
def load_data(file_path):
    """ Load dataset from a CSV file. """
    try:
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        raise Exception(f"Error loading file: {e}")


# Main execution function
def main():
    file_path = r'\\win-g12\\ResearchWorks\\Interns\\jing.yu\\python file\\Project_1\\sc_ema_merged_output.csv'
    data = load_data(file_path)

    required_columns = ['date', 'code', 'HV5', 'HV10', 'HV15', 'HV20', 'HV25', 'HV30', 'ATM Volatility', 'Futures']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")

    threshold = 0.02
    initial_position = 0
    trade_size = 1

    data = generate_signals_with_trigger_new(data, threshold, initial_position)
    data = calculate_strategy_returns(data)
    data = calculate_cumulative_returns(data)

    output_file_path = r'\\win-g12\\ResearchWorks\\Interns\\jing.yu\\python file\\Project_1\\sc_2ema_result_new.csv'
    data.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}!")


if __name__ == "__main__":
    main()