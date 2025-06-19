#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:40:57 2025

@author: fredrik
"""



def what_is_daily_rainfal(totdata_list1, totdata_list2=False,
                          how_many_over=False, how_many_over2=False, 
                          how_many_over3=False, info=False):
    """ 
    This function takes a totdata_list and gives you a list of how much 
    rainfall is experienced during a 24 h period. 
    """
    daily_rain_list = [] 
    daily_rain = 0
    for event in totdata_list1: # Loop through all events
        for h in range(len(event.T)): # Loop through each hour
            hour = event[0][h]          
            hourly_rain = float(event[5][h])
            daily_rain += hourly_rain # Sum up for rainperiod
        
            if hour % 24 == 0: # If divisible by 24, than it is a day
                daily_rain_list.append(daily_rain)
                daily_rain = 0
    
    if totdata_list2: # If two lists are given, check other one the same way
        daily_rain = 0
        for event in totdata_list2: # Loop through all events
            for h in range(len(event.T)):
                hour = event[0][h]
                hourly_rain = float(event[5][h])
                daily_rain += hourly_rain
            
                if hour % 24 ==0:
                    daily_rain_list.append(daily_rain)
                    daily_rain = 0

    if how_many_over: # Print how many days a value is exceeded or not
        count = 0
        for rainfall in daily_rain_list:
            if rainfall >= how_many_over:
                count += 1
        per = np.round(100*count/len(daily_rain_list),3) # Calculate percentage
        print(f"{count} days exceeded {how_many_over}mm/d which is {per}%")
   
    if how_many_over2: # Can be done for more limits
        count = 0
        for rainfall in daily_rain_list:
            if rainfall >= how_many_over2:
                count += 1
        per = np.round(100*count/len(daily_rain_list),3)
        print(f"{count} days exceeded {how_many_over2}mm/d which is {per}%")
    
    if how_many_over3: # Can be done for more limits
        count = 0
        for rainfall in daily_rain_list:
            if rainfall >= how_many_over3:
                count += 1
        per = np.round(100*count/len(daily_rain_list),3)
        print(f"{count} days exceeded {how_many_over3}mm/d which is {per}%")
    
    if info:
        decimals = 2
        mean = np.round(np.nanmean(daily_rain_list), decimals)
        std = np.round(np.nanstd(daily_rain_list), decimals)
        print(f"A total of {len(daily_rain_list)} das were used:")
        print(f"The mean rainfall was {mean} with a standrad deviation of {std}")

       
    daily_rain_list_rounded = np.round(daily_rain_list,4) # Round values
    daily_rain_list_sorted = daily_rain_list_rounded.sort() # Sort
    
    return daily_rain_list # Return list of 24h rainfall. 



def is_daily_rainfall_exceeded(totdata_list1, totdata_list2=None, limit1=None,
                               limit2=None):
    """
    Checks how many events exceed a daily rainfall threshold.
    Works on one or two datasets of events.
    """
    def count_exceeding_events(data_list, limit):
        count = 0
        for event in data_list:
            daily_rain = 0
            for h in range(len(event[0])):  # event[0] = hour, event[5] = rain
                hourly_rain = float(event[5][h])
                daily_rain += hourly_rain
                if (h + 1) % 24 == 0:
                    if daily_rain >= limit:
                        count += 1
                        break  # Only count one day per event
                    daily_rain = 0
        return count

    if limit1 is None:
        raise ValueError("limit1 must be specified.")

    counter1 = count_exceeding_events(totdata_list1, limit1)
    total_events = len(totdata_list1)

    if totdata_list2:
        counter1 += count_exceeding_events(totdata_list2, limit1)
        total_events += len(totdata_list2)
        
 

    per1 = np.round(100 * counter1 / total_events, 2)

    print(f"Total number of events: {total_events}")
    print(f"{counter1} events exceeded {limit1} mm/day, which is {per1}%")

    return counter1, per1

    


def what_is_blocking_rainfal(totdata_list1, totdata_list2=False,
                             how_many_over=False, how_many_over2=False, 
                             how_many_over3=False, info=False):
    blocking_rain_list = [] 
    for event in totdata_list1: # Loop through all events
        event_rain = 0
        for h in range(len(event.T)): # Loop through each hour
            hourly_rain = float(event[5][h])
            event_rain += hourly_rain # Sum up for rainperiod
        blocking_rain_list.append(event_rain)
    
    if totdata_list2: # If two lists are given, check other one the same way
        for event in totdata_list1: # Loop through all events
            event_rain = 0
            for h in range(len(event.T)): # Loop through each hour
                hourly_rain = float(event[5][h])
                event_rain += hourly_rain # Sum up for rainperiod
            blocking_rain_list.append(event_rain)
            
    blocking_rain_list_rounded = np.round(blocking_rain_list,4) # Round values
    blocking_rain_list_sorted = blocking_rain_list_rounded.sort() # Sort

    return blocking_rain_list_rounded # Return list of 24h rainfall. 
    

