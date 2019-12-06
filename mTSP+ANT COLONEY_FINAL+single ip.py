# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:52:10 2019

@author: PRASHANT
"""

from gurobipy import*
import os
import xlrd
import numpy as np
from numpy import inf
from scipy import spatial
import numpy
from sklearn.metrics.pairwise import euclidean_distances
import math
import time

iteration = 100
n_ants =16
n_facility = 16        ####### 1 PLUS ACTUAL NO OF FACILITES
n_cust=35
e=0.5
alpha=2
break_limit=10
beta=2
maximum_distance=50
n_salesman=3
nc=6            ###ALSO CORRECT LINE 89
facility_cordinate={}
cust_cordinate={}

demand = np.ones((n_cust))

book = xlrd.open_workbook(os.path.join("1.xlsx"))
sh = book.sheet_by_name("Sheet1")
i = 1
l=0
for i in range(1,n_facility+1):  
    sp = sh.cell_value(i,1)
    sp2 = sh.cell_value(i,2)
    sp1=(sp,sp2)
    facility_cordinate[l]=(sp1)
    l=l+1
j=0
for i in range(n_facility+1,n_cust+1+n_facility):  
    sp = sh.cell_value(i,1)
    sp2 = sh.cell_value(i,2)
    sp1=(sp,sp2)
    cust_cordinate[j]=sp1  
    j=j+1
    
def calculate_dist(x1, x2):
    eudistance = spatial.distance.euclidean(x1, x2)
    return(eudistance)

f_dist=[]
     
for i in facility_cordinate:
    facility_dist=[]
#    a=facility_cordinate[i]
    for j in facility_cordinate:
        facility_dist.append(calculate_dist(facility_cordinate[i],facility_cordinate[j]))
    f_dist.append(facility_dist)
fac_dist=np.array(f_dist)    
customer_dist={}
for i in facility_cordinate:
    if i!=0:
        for j in cust_cordinate:
            customer_dist[i,j]=calculate_dist(facility_cordinate[i],cust_cordinate[j])

abc={}
for i in range(1,n_facility+1):
    j = 3
    xyz=[]
    while True:
        try:
            
            sp = sh.cell_value(i,j)
            xyz.append(sp)
            
            j = j + 1
            
        except IndexError:
            break
    abc[i-1]=xyz
    
final_aij=[]
for i in range(n_facility):
    a_ij=[]
    for j in range(n_facility,n_facility+n_cust):
        if j in abc[i]:
            a_ij.append(1)
        else:
            a_ij.append(0)
    final_aij.append(a_ij)
cust_dist=np.array(final_aij)
#
###
zx=[]

for i in range(n_cust):
    zx.append(i)
zx.sort()
#################################################################
################################################################
demand = demand[:,np.newaxis]
pheromne = 0.15*np.ones((n_ants,n_facility))
route_name=[]
for i in range(1,n_salesman+1):
    route_name.append("route"+str(i))
#print(route_name)

for i in range(len(route_name)):
    a= np.ones((n_ants,n_facility))
    route_name[i]=a       

demand_satisfied_list=[]

for i in range(1,n_salesman+1):
    demand_satisfied_list.append("demand_satisfied_list"+str(i))
#print(demand_satisfied_list)

for i in range(len(demand_satisfied_list)):
    a= []
    demand_satisfied_list[i]=a
    
overall_best_route=[]
for i in range(1,n_salesman+1):
    overall_best_route.append("overall_best_route"+str(i))

for i in range(len(overall_best_route)):
    a= []
    overall_best_route[i]=a

demand_satisfied_array=[]
for i in range(1,n_salesman+1):
    demand_satisfied_array.append("demand_satisfied_array"+str(i))
    
route_opt=[]
for i in range(1,n_salesman+1):
    route_opt.append("route_opt"+str(i))
    
#pheromne=[]
#for i in range(1,n_salesman+1):
#    pheromne.append("pheromne"+str(i))
#
#for i in range(len(pheromne)):
#    a=  0.15*np.ones((n_ants,n_facility))
#    pheromne[i]=a

#route = np.ones((n_ants,n_facility))



no_of_cust_covered = np.zeros((1,n_facility))
dem_sat_array=np.zeros((1,n_facility))



for num1 in range(n_facility):
    s=0
    dem_sat=0
    for num2 in range(n_cust):
        if cust_dist[num1,num2]==1:
                    s+=1
                    dem_sat+=demand[num2]
    no_of_cust_covered[0,num1]=s
    dem_sat_array[0,num1]=dem_sat

factor=1/fac_dist    
factor[factor==inf]=0
visibility=factor*dem_sat_array

overall_max_satisfied_cust=0
cost_matrix=np.zeros((iteration,1))
facility_name=[]
for i in range(1,n_facility):
    facility_name.append(i)
start_time = time.time()

for ite in range(iteration):             #iteration
    
#    for Q in route_name:
#        
#        W=np.ones((n_ants,n_facility))
#        Q=W
    for Q in range((len(route_name))):
        W=np.ones((n_ants,n_facility))
        route_name[Q]=W
    
#    route = np.ones((n_ants,n_facility))#####*******
#    demand_satisfied_list=[]
#    print("iteration =",ite)
#    print("demand satisfied list 1",demand_satisfied_list[0])
#    print("demand satisfied list 2",demand_satisfied_list[1])
    
    for Z in range(len( demand_satisfied_list)):
        a=[]
        demand_satisfied_list[Z]=a
#    print("AFTER")
#    print("demand satisfied list 1",demand_satisfied_list[0])
#    print("demand satisfied list 2",demand_satisfied_list[1])
    
    for i in range(n_ants):              # no of ants
        
        temp_visibility = np.array(visibility)
        demand_satisfied=0
        distance_covered=0
        covered_facilities=[]
        unsatisfied_cust=[]
        satisfied_cust=[]
        total_satisfied_cust=[]
        unique_satisfied_cust=set()
        temp_no_of_cust_covered=np.array(no_of_cust_covered)
        temp_dem_sat_array=np.array(dem_sat_array)
        
        temp_cust_distance=np.array(cust_dist)


        for u in range(n_salesman):
            if u>0:
                bahubali=[]             #to store facility which satisfies 0 customers
                for n1 in range(n_facility):
                    s=0
                    dem_sat=0
                    for n2 in range(n_cust):
                        if temp_cust_distance[n1,n2]==1:
                            s+=1
                            dem_sat+=demand[n2]
                            
                    temp_no_of_cust_covered[0,n1]=s
                    temp_dem_sat_array[0,n1]=dem_sat
                    
                for a1 in range(n_facility):
                    if temp_dem_sat_array[0,a1]==0:
                       
                        bahubali.append(a1)

                temp_visibility=factor*temp_dem_sat_array
                for b1 in bahubali:
                    temp_visibility[:,b1]=0
            
            satisfied_cust=[]
                
        
        
            for j in range(n_facility-1):
                fac=zx[:]
                
#                if (distance_covered<maximum_distance):
                if distance_covered<maximum_distance and len(unique_satisfied_cust)!=n_cust:
                    if j>0:
                        bahubali=[]             #to store facility which satisfies 0 customers
                        for n1 in range(n_facility):
                            s=0
                            dem_sat=0
                            for n2 in range(n_cust):
                                if temp_cust_distance[n1,n2]==1:
                                    s+=1
                                    dem_sat+=demand[n2]
                                    
                            temp_no_of_cust_covered[0,n1]=s
                            temp_dem_sat_array[0,n1]=dem_sat
                            
                        for a1 in range(n_facility):
                            if temp_dem_sat_array[0,a1]==0:
                               
                                bahubali.append(a1)
    
                        temp_visibility=factor*temp_dem_sat_array
                        for b1 in bahubali:
                            temp_visibility[:,b1]=0
    
                            
                    demand_satisfied=0
                    distance_covered=0
                    combine_feature = np.zeros(n_facility)
                    cum_prob = np.zeros(n_facility)
                    cur_loc = int(route_name[u][i,j]-1)
                    temp_visibility[:,cur_loc] = 0
                    p_feature = np.power(pheromne[cur_loc,:],beta)
                    v_feature = np.power(temp_visibility[cur_loc,:],alpha)
                    p_feature = p_feature[:,np.newaxis]
                    v_feature = v_feature[:,np.newaxis]
                    combine_feature = np.multiply(p_feature,v_feature)
                    total = np.sum(combine_feature)
                    if total==0:
                        total=1 
                    probs = combine_feature/total
                    cum_prob = np.cumsum(probs)
                    r = np.random.random_sample()
                    if np.all(cum_prob==0):
                        facility=1
                        route_name[u][i,j+1] = facility
#                        print("facility- ",facility)
#                        covered_facilities.append(facility-1)
                        
                        break
                    else:
                    
                        facility = np.nonzero(cum_prob>r)[0][0]+1
#                        print("facility- ",facility)
                    
#                    facility = np.nonzero(cum_prob>r)[0][0]+1
    
                    route_name[u][i,j+1] = facility
                    for v in range(n_facility-1):
                        distance_covered= distance_covered+fac_dist[int(route_name[u][i,v])-1,int(route_name[u][i,v+1])-1]
#                    print("dist covered- ",distance_covered)
#                    if distance_covered<maximum_distance:
                    if distance_covered<maximum_distance and len(unique_satisfied_cust)!=n_cust:
                        for k in range(n_cust):
                            if (temp_cust_distance[facility-1,k]==1):
                               satisfied_cust.append(k)
                        for h in satisfied_cust:
                           total_satisfied_cust.append(h)
                        unique_satisfied_cust=set(total_satisfied_cust)
                           
                        covered_facilities.append(facility-1)
                               
                        for g1 in satisfied_cust:
                            fac.remove(g1)
                        unsatisfied_cust=fac
        
                        for b in satisfied_cust:
                            demand_satisfied+=np.sum(demand[b,0])
    
                        for a in satisfied_cust:
                            temp_cust_distance[:,a]=0
                    else:
                        distance_covered=0
#                        demand_satisfied=0
                        route_name[u][i,j+1]=1
                        for v in range(n_facility-1):
                            distance_covered= distance_covered+fac_dist[int(route_name[u][i,v])-1,int(route_name[u][i,v+1])-1]
                        for b in satisfied_cust:
                            demand_satisfied+=np.sum(demand[b,0])
                            
                        demand_satisfied_list[u].append(demand_satisfied)
                        
    
                        break
    
    ################### WE CAN CHECK OTHER COMBINATIONS OTHER THAN THE CHOOSEN FACILITY SO THAT WE MAY GET A FACILITY WHICH IF INCLUDED, THE DISTANCE TRAVELLED IS STILL < MAXIMUM DIST. IF NO SUCH COMB IS AVAILABLE THEN BREAK.
    #                    break
                    
                elif distance_covered<maximum_distance and len(unique_satisfied_cust)==n_cust:
                        distance_covered=0
                        demand_satisfied=0
                        route_name[u][i,j+1]=1
                        for v in range(n_facility-1):
                            distance_covered= distance_covered+fac_dist[int(route_name[u][i,v])-1,int(route_name[u][i,v+1])-1]
                        for b in satisfied_cust:
                            demand_satisfied+=np.sum(demand[b,0])
                            
                        demand_satisfied_list[u].append(demand_satisfied)
                        
    
                        break
                else:
                      break
##########    for i in route_name:
###########        print(i)
##########    for i in demand_satisfied_list:
###########        print(i)
                    
                
#    route_opt = np.array(route)               #intializing optimal route
    for i in range(len(route_name)):
        route_opt[i]=np.array(route_name[i])
#    demand_satisfied_array=np.array(demand_satisfied_list)
    for i in range(len(demand_satisfied_list)):
        demand_satisfied_array[i]=np.array(demand_satisfied_list[i])
        
    
#    satisfied_demand_max_loc = np.argmax(demand_satisfied_array)
#
#    max_satisfied_cust = demand_satisfied_array[satisfied_demand_max_loc]         #finging min of dist_cost
    
        
    
    
    overall_demand_satisfied_list=[sum(x) for x in zip(*demand_satisfied_list)]
    overall_demand_satisfied_array=np.array(overall_demand_satisfied_list)
    
    if overall_demand_satisfied_array.size==0:
        dist_max_loc=0
        dist_max_cost=0
    else:
        satisfied_demand_max_loc = np.argmax(overall_demand_satisfied_array)
        max_satisfied_cust = overall_demand_satisfied_array[satisfied_demand_max_loc]

    cost_matrix[ite]=overall_max_satisfied_cust               ##BREAKING CRITERIA
    if ite>break_limit:
        out=0
        for v in range(ite,ite-break_limit,-1):
            if cost_matrix[v]==cost_matrix[v-1]:
                out+=1
        if out==break_limit:
            break
         

    
############INITIALISING MIN AND MAX PHEROMONE LEVEL    
    if max_satisfied_cust > overall_max_satisfied_cust:
        overall_max_satisfied_cust=max_satisfied_cust
        
#        print("**********************************")
#        print("**********************************")
        
        for A in range(len(overall_best_route)):
            overall_best_route[A]=route_name[A][satisfied_demand_max_loc]
#        print(overall_best_route)
                
#        overall_best_route=route[satisfied_demand_max_loc]
        maximum_pheromne=1/(e* overall_max_satisfied_cust)
        minimum_pheromne=maximum_pheromne/5
#               
    print("best sol in iteration" , ite," is ",overall_max_satisfied_cust)     
#    best_route = route[satisfied_demand_max_loc,:]
#
######LOCAL PHEROMNE UPDATE    
    pheromne = (1-e)*pheromne
    
    for c in range(len(overall_demand_satisfied_array)):
        for v in range(n_facility-1):
            dt = overall_demand_satisfied_array[c]/demand.sum(axis=0)
            for B in range(n_salesman):
                pheromne[int(route_opt[B][c,v])-1,int(route_opt[B][c,v+1])-1] = pheromne[int(route_opt[B][c,v])-1,int(route_opt[B][c,v+1])-1] + dt
#    
#    
######GLOBAL PHEROMONE UPDATE 
    for B in range(n_salesman):            
        for c in range(n_facility-1):        
            pheromne[int(overall_best_route[B][c])-1,int(overall_best_route[B][c+1])-1]=pheromne[int(overall_best_route[B][c])-1,int(overall_best_route[B][c+1])-1]+dt

###### MAXIMUM AND MIN LIMIT ON PHEROMNE
    for B in range(n_salesman):        
        for c in range(n_ants):
            for v in range(n_facility):    
                if pheromne[c,v]<minimum_pheromne:
                    pheromne[c,v]=minimum_pheromne
                if pheromne[c,v]>maximum_pheromne:
                    pheromne[c,v]=maximum_pheromne
#
#  
#    
final_dist=0
for C in range(n_salesman):
    for v in range(n_facility-1):
        final_dist+= fac_dist[int(overall_best_route[C][v])-1,int(overall_best_route[C][v+1])-1]
    
total_time=time.time() - start_time    
#print('route of all the ants at the end :')
#print(route_opt)
print()
print('best path :',overall_best_route)
#print('cost of the best path',int(dist_min_cost[0]) + fac_dist[int(best_route[-2])-1,0])  
print('maximum demand satisfied =',overall_max_satisfied_cust)        
print("total distance covered =",final_dist)
print("total time taken =",total_time) 
#print(len(unique_satisfied_cust))   
    