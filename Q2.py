#!/usr/bin/env python
# coding: utf-8

# # Question 2: Vaccine Distribution Modelling

# In[ ]:

import streamlit as st
import constraint
import math 

#predefine data for every state
max_capacity = [5000,10000,7500,8500,9500]
total_citizen = [565790,514544,889038,1153093,690884 ]
citizen_smaller_35 = [115900,100450,223400,269300,221100]
citizen_between_35_60 = [434890,378860,643320,859900,450500]
citizen_greater_60 = [15000,35234,22318,23893,19284]
total_CR1 = [20,30,22,16,19]
total_CR2 = [15,16,15,16,10]
total_CR3 = [10,15,11,16,20]
total_CR4 = [21,10,12,15,15]
total_CR5 = [5,2,3,1,1]
state = 1

for i in range(5): 

    #Get the number of days it will take to vaccinate everybody
    vac_a_day = math.ceil(citizen_greater_60[i]/max_capacity[i])
    vac_b_day = math.ceil(citizen_between_35_60[i]/max_capacity[i])
    vac_c_day = math.ceil(citizen_smaller_35[i]/max_capacity[i])
    total_day = vac_a_day + vac_b_day + vac_c_day

    problem = constraint.Problem()

    problem.addVariable('CR1', range(total_CR1[i]+1))  
    problem.addVariable('CR2', range(total_CR2[i]+1))  
    problem.addVariable('CR3', range(total_CR3[i]+1)) 
    problem.addVariable('CR4', range(total_CR4[i]+1)) 
    problem.addVariable('CR5', range(total_CR5[i]+1))  


    # We have different number of maximum vaccine for each state per day
    def capacity_constraint(a, b, c, d, e):  
        if (a*200 + b*500 + c*1000 + d*2500 + e*4000) >= max_capacity[i]:
            return True


    problem.addConstraint(capacity_constraint,['CR1','CR2','CR3','CR4','CR5'])

    rental = 999999999999999 
    solution_found = {}  
    solutions = problem.getSolutions()

    # Get the conditions for the minimum rental

    for s in solutions:
       current_rental = s['CR1']*100 + s['CR2']*250 + s['CR3']*500 + s['CR4']*800 + s['CR5']*1200
       current_capacity = s['CR1']*200 + s['CR2']*500 + s['CR3']*1000 + s['CR4']*2500 + s['CR5']*4000
       if current_rental < rental and current_capacity == max_capacity[i]:
            rental = current_rental
            solution_found = s

    st.write("""
    In State {}    
    Everyday, we'll rent:  
    {} CR-1
    {} CR-2
    {} CR-3
    {} CR-4
    {} CR-5

    The minimum rental per day is {} with maximum capacity of {}

    All citizen will be vaccinated in {} days.

    By priotizing citizen age < 35, followed by 35 < age < 65, then age > 65\n
    We will only distribute Vaccine C for the first {} days\n
    We will only distribute Vaccine B for the next {} days\n
    We will only distribute Vaccine A for the next {} days
    """.format(state, solution_found['CR1'], solution_found['CR2'], solution_found['CR3'], solution_found['CR4'],solution_found['CR5'],rental,max_capacity[i],total_day,vac_c_day,vac_b_day,vac_a_day))
    state += 1 


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=158f0e5f-0c09-4da4-a8e2-16b96f03a497' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
