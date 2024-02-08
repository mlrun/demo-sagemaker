import pandas as pd
import datetime

#from datetime import datetime, timedelta

def update_timestamps(data):    
    
    # Step 3: Get the current time
    now = pd.Timestamp(datetime.datetime.now())
    
    # Step 4: Calculate the time difference
    time_difference = now - data['timestamp'].iloc[-1]
    
    # Step 5: Adjust all timestamps
    data['timestamp'] = data['timestamp'] + time_difference
    
    # Display the adjusted DataFrame
    return data


# # Function that updates the timestamps so each transaction category has rows with timestamps from the last 5 days (2 per day)
# def update_timestamps(data):
#     # Get today's date
#     today = datetime.today()

#     # Calculate the dates for the last 5 days
#     last_5_days = [today - timedelta(days=i) for i in range(4, -1, -1)]  # Reverse for chronological order

#     # Extract year, month, and day from each date object
#     years = [d.year for d in last_5_days]
#     months = [d.month for d in last_5_days]
#     days = [d.day for d in last_5_days]

#     hours = [10, 15]

#     # Create a list of timestamps of the last 5 days, 2 timestamps per day.
#     times = []
#     for year, month, day in zip(years, months, days):
#         for hour in hours:
#             times.append(datetime(year, month, day, hour))

#     # Iterate over each transaction category
#     for i in range(len(data["transaction_category"].unique())):
#         # Extract all the rows for each category
#         category_data = data[data['transaction_category'] == str(i)]

#         # Ensure timestamp is a datetime object
#         pd.to_datetime(category_data.timestamp)

#         # Sort DataFrame by timestamp in descending order
#         category_data_sorted = category_data.sort_values(by='timestamp', ascending=False)

#         # Select the latest rows and update their timestamp
#         latest_rows = category_data_sorted.head(len(times))
#         latest_rows.loc[:, 'timestamp'] = times

#         # Update the initial dataframe to include those updated rows
#         data.update(latest_rows)

#         data.sort_values(["transaction_category", "timestamp"], inplace=True)


#     return data