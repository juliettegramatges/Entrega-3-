import os
import math
import numpy as np
import multiprocessing as mp
import os.path
from time import time
import random
import pandas as pd
import csv


def loadDistances(fname):
    # Obtener ruta absoluta del archivo
    fname = os.path.abspath(fname)
    # Leer los pares de ciudades y distancias
    cities_set = set()
    distances_dict = {}
    with open(fname, mode="r") as file:
        reader = csv.reader(file)
        
        for row in reader:
            city1, city2 = row[0], row[1]
            distance = int(row[2])
            
            # Añadir ciudades al set
            cities_set.update([city1, city2])
            
            # Guardar la distancia en un diccionario usando tuplas de ciudades como clave
            distances_dict[(city1, city2)] = distance
            distances_dict[(city2, city1)] = distance  # Simetría de la distancia

    # Crear una lista de ciudades únicas y ordenar
    cities = sorted(cities_set)
    
    # Crear una matriz cuadrada con ceros
    num_cities = len(cities)
    distance_matrix = [[0] * num_cities for _ in range(num_cities)]

    # Llenar la matriz usando el diccionario de distancias
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:  # Evitar la distancia de una ciudad a sí misma
                distance_matrix[i][j] = distances_dict.get((city1, city2), 0)

    return distance_matrix, cities

class Config:
    def __init__(self, vuelo, scenario):
        # Si 'vuelo' es un diccionario, asegúrate de que los valores se extraigan correctamente
        self.nodos = [vuelo['origin']] + vuelo['destinations'].split(", ")
        
        self.weiCap = 0
        self.volCap = 0
        self.numNodes = len(self.nodos)
        self.Sce      = scenario
        self.plane_type = vuelo['Plane Type']
        self.flight_type = vuelo['Flight Type']
        self.vol      = vuelo['Volume (cubic meters)']
        self.wei = vuelo['Weight (kg)']

        self.numPallets = 6 if self.flight_type == 'Narrow-Body' else 10
        self.payload    = 26_000
        self.maxTorque  = 26_000 * 0.556
        self.kmCost     = vuelo['cost_per_km']

class Pallet(object):
    def __init__(self, id, d, v, w, numNodes):
        self.ID = id
        self.D  = d  # centroid distance to CG
        self.V  = v  # volume limite
        self.W  = w  # weight limite
        self.Dest = np.full(numNodes, -1) # nodes IDs indexed by "k"
        self.PCW = 0 # pallet current weight
        self.PCV = 0.
        self.PCS = 0.
        self.w = 2.15 # width 84"
        self.d = 2.65 # depth 104"
        self.h = 1.5 * self.V /(self.w * self.d) # height

    
    
    def reset(self, numNodes):
        self.Dest = np.full(numNodes, -1)
        self.PCW = 140
        self.PCV = 0.
        self.PCS = 0.

    def popItem(self, item, nodeTorque, solDict, N, itemsDict): # pop an item from this pallet
        self.PCW -= item.W
        self.PCV -= item.V
        self.PCS -= item.S

        nodeTorque.value -= float(item.W) * float(self.D)
        i = self.ID
        j = item.ID 
        solDict["solMatrix"][N*i+j] = 0 # solution array
        itemsDict["mpItems"][j]     = 0 # to control items inclusion

    def putItem(self, item, nodeTorque, solDict, N, itemsDict, lock): # put an item in this pallet
        self.PCW += item.W
        self.PCV += item.V
        self.PCS += item.S

        with lock: # manage race condition
            nodeTorque.value += float(item.W) * float(self.D)
            i = self.ID
            j = item.ID 
            solDict["solMatrix"][N*i+j] = 1 # solution array
            itemsDict["mpItems"][j]     = 1 # to control items inclusion

    def putConsol(self, consol): # put a consolidated in this pallet
        self.PCW += consol.W
        self.PCV += consol.V
        self.PCS += consol.S
            
    def isFeasible(self, item, threshold, k, nodeTorque, cfg, itemsDict, lock, ts=1.0): # check constraints

        feasible = True

        if feasible and item.To != self.Dest[k]:
            feasible = False
        # Verifica si el destino (item.To) coincide con el destino del pallet (self.Dest[k]). 
        # Si no coincide, feasible se establece en False.

        if feasible and self.PCV + item.V > self.V * threshold:
            feasible = False
        # Comprueba si el volumen total después de añadir el ítem excede la capacidad
        #  de volumen disponible del pallet (self.V * threshold). Si es así, feasible se establece en False.

        if feasible and self.PCW + item.W > self.W:
            feasible = False
        # Verifica que el peso total después de añadir el ítem no supere el límite 
        # de peso del pallet (self.W). Si lo supera, feasible se establece en False.

        if feasible:
            with lock: # ask for a lock to control race condition
                j = item.ID
                # if itemsDict["mpItems"][j] > 0: Este condicional verifica si el ítem ya ha sido asignado 
                # (si itemsDict["mpItems"][j] es mayor que cero, el ítem ya fue colocado en otro lugar).
                if itemsDict["mpItems"][j] > 0: # if it is greater than zero it is already allocated
                    feasible = False


                if feasible:
                    deltaTau = float(item.W) * float(self.D)
                    newTorque = nodeTorque.value + deltaTau
                    """
                    Infeasible:
                    If the torque is increasing positively and is bigger  than  maxTorque
                    If the torque is increasing negatively and is smaller than -maxTorque
                    """
                    # hay que elegir un valor máximo de torque, para que se pueda balancear después.
                    if  abs(nodeTorque.value) < abs(newTorque) and cfg.maxTorque*ts < abs(newTorque):
                        feasible = False                       
        return feasible

def loadPallets(cfg):
    """
    Load pallets attributes based on aircraft size
    """

    dists = []
    vol = 0
    wei = 0

    if cfg.flight_type == "Passenger" & cfg.plane_type == "Narrow-Body":
        vol = cfg.vol/6
        wei = 10**100
        dists = [2.5,1.5,0.5,-0.5,-1.5,-2.5] # distances of pallets centroids to the center of gravity
    
    if cfg.flight_type == "Cargo" & cfg.plane_type == "Narrow-Body":
        vol = cfg.vol/6
        wei = 10**100
        dists = [7.5,4.5,1.5,-1.5,-4.5,-7.5] # distances of pallets centroids to the center of gravity

    if cfg.flight_type == "Passenger" & cfg.plane_type == "Wide-Body":
        vol = cfg.vol/10
        wei = 10**100
        dists = [5.85,4.55,3.25,1.95,0.65,-0.65,-1.95,-3.25,-4.55,-5.85] # distances of pallets centroids to the center of gravity
    
    if cfg.flight_type == "Cargo" & cfg.plane_type == "Wide-Body":
        vol = cfg.vol/10
        wei = 10**100
        dists = [13.5,10.5,7.5,4.5,1.5,-1.5,-4.5,-7.5,-10.5,-13.5] # distances of pallets centroids to the center of gravity


    pallets = []
    id = 0

    for d in dists:
        pallets.append( Pallet(id, d, vol, wei, cfg.numNodes) )
        id += 1
   
    return pallets, dists[0] # ramp door distance from CG


class Tour(object):
    def __init__(self, nodes, cost):
        self.nodes = nodes
        self.cost  = cost # sum of legs costs plus CG deviation costs
        self.score   = 0.0 # sum of nodes scores
        self.elapsed  = 0 # seconds no 3D packing
        self.elapsed2 = 0 # seconds with 3D packing
        self.numOpts = 0 # sum of nodes eventual optima solutions
        self.AvgVol  = 0.0 # average ocupation rate
        self.AvgTorque  = 0.0

def factorial(x):
    result = 1
    for i in range(x):
        result *= i+1
    return result

def permutations(n):
    fac = factorial(n)
    a = np.zeros((fac, n), np.uint32) # no jit
    f = 1
    for m in np.arange(2, n+1):
        b = a[:f, n-m+1:]      # the block of permutations of range(m-1)
        for i in np.arange(1, m):
            a[i*f:(i+1)*f, n-m] = i
            a[i*f:(i+1)*f, n-m+1:] = b + (b >= i)
        b += 1
        f *= m
    return a

# Devuelve la matriz a que contiene todas las permutaciones posibles de un conjunto de n elementos.

class Node(object):
    def __init__(self, id, icao, diccionario):
        self.ID   = id
        self.tLim = 1 # 1s
        self.Vol  = 0.0
        for key, value in diccionario.items():
            if id == value:
                self.ICAO = key
            break 



class Item(object):
    """
    A candidate item "j" to be loaded on a pallet "i" if X_ij == 1
    """
    # consol           -1, -2, 0, 0, 0., -1, -1
    def __init__(self, id, p, w, s, v, h, d, frm, to):

        self.w = w
        self.d = d
        self.h = h

        self.ID = id
        self.P  = p  # -1 if an item, -2 if a consollidated, or pallet ID. 
        # ID del palé en el que el ítem podría ir, o valores especiales como -1 (ítem independiente) y -2 (ítem consolidado).
        self.w  = w  # weight
        self.s  = s  # score
        self.V  = v  # volume
        self.Frm = frm  # origin node ID
        self.To = to # destination node ID
        self.Attr = 0.0



def getTours(num, costs, threshold, aeropuertos):

    p = permutations(num)

    #+2: the base before and after the permutation
    # Se crea una lista de listas, toursInt, que tiene el mismo número de filas que el número de permutaciones (p), 
    # y cada fila tiene dos columnas adicionales que se utilizan para representar el recorrido (con un valor inicial de 0).
    
    toursInt = [
        [0 for _ in np.arange(len(p[0])+2)] for _ in np.arange(len(p))
        ]

    # define the core of the tours. Se recorren todas las permutaciones generadas y se llenan las listas dentro de toursInt 
    # con los valores correspondientes, ajustando el índice (sumando 1) ya que se está trabajando con índices que empiezan desde 1.
    for i, row in enumerate(p):
        for j, col in enumerate(row):
            toursInt[i][j+1] = col+1


    tours = [None for _ in np.arange(len(toursInt))]
    minCost = 9999999999999.
    maxCost = 0.  

    # Se crea la lista tours que almacenará los objetos Tour (probablemente definidos en otro lugar del código).
    # Se inicializan las variables minCost y maxCost para seguir el costo mínimo y máximo de los tours. 

    for i, tour in enumerate(toursInt):
        nodes = []
        cost = 0.
      
        for j, nid in enumerate(tour):
            n = Node(nid, 0., aeropuertos)
            nodes.append( n )

            if j>0:
                frm = nodes[j-1].ID
                to  = nodes[j].ID
                if frm < len(costs) and to < len(costs[frm]):
                    cost += costs[frm][to]

            if j == len(tour[:-1])-1: # the last node
                frm = nodes[j].ID
                to  = 0
                if frm < len(costs) and to < len(costs[frm]):
                    cost += costs[frm][to]                

        if cost < minCost:
            minCost = cost

        if cost > maxCost:
            maxCost = cost

        tours[i] = Tour(nodes, cost)

    if len(tours) <= 24:
        return tours

    tours2 = []
    for i, t in enumerate(tours):
        if t.cost <= minCost + (maxCost - minCost) * threshold:
            tours2.append(t)
    tours    = None
    toursInt = None

    return tours2


# used in sequential mode
def loadNodeItems(scenario, instance, node, unatended, fname, aeropuertos): # unatended, future nodes
    """
    Load this node to unnatended items attributes
    Items destined outside the rest of the flight plan will not be loaded (13 and 14).
    """

    reader = open(fname, "r")
    lines = reader.readlines() 

    items = []
    id = 0

    # Para cada línea leída:
    # Se asignan las columnas a variables: peso (w), puntuación (s), volumen (v), nodo de origen (frm), y nodo de destino (to).
    # Si el nodo de origen coincide con el nodo actual (node.ID) y el nodo de destino está en la lista de nodos no atendidos (unatended), 
    # el item se agrega al nodo y su volumen se incrementa en v. Luego, el item es añadido a la lista items.

    # veo la demanda de cada nodo
    try:
        for line in lines:
            cols = line.split().split(",")
            w   =   int(cols[4]) # width
            s   =   int(cols[6]) # benefit
            h   =   int(cols[3]) # height
            d   =   int(cols[2]) # wide
            v   =   int(cols[2])*int(cols[3])*int(cols[4])
            frm =   aeropuertos[cols[0]]
            to  =   aeropuertos[cols[2]]
            if frm == node.ID and to in unatended:
                node.Vol += v       
                items.append( Item(id, -1, w, s, v, h, d, frm, to) ) # P:-1 item, -2: consolidated
                id += 1

    finally:
        reader.close()  
          
    items.sort(key=lambda x: x.S/x.V, reverse=True)
    id = 0

    # prepare for ACO

    # Este código asigna una "atractividad" a cada elemento en items basada en la relación de su puntaje y volumen, 
    # ajustada por el valor de bestAttr. Luego asigna un ID único a cada elemento, incrementando id en cada iteración. 
    # Esto se utiliza en ACO para favorecer las soluciones que consideran la relación más atractiva de S/V.

    # proceso de Optimización por Colonia de Hormigas (Ant Colony Optimization, ACO) al calcular una medida de "atractividad" 
    # (attractiveness) para cada elemento en items

    if len(items) > 0:

        bestAttr = items[0].S / items[0].V # the first item has the best attractiveness
        # avgAttr = 0.0
        for i, it in enumerate(items):
            items[i].Attr = 3. * (it.S/it.V) / bestAttr # 4: to make the average around 0.5
            # avgAttr += items[i].Attr
            items[i].ID = id
            id += 1

    # avgAttr /= len(items)
    # print(f"avgAttr = {avgAttr:.3f}")

    return items, node

def setPalletsDestinations(items, pallets, nodes, k, L_k):

    vols  = [0]*len(nodes)
    max   = 0 # node with maximum volume demand
    total = 0

    # all items from all nodes
    for it in items:
        # the items from this node
        if it.Frm == nodes[k].ID and it.P == -1:
            d = it.To
            if d in L_k:
                vols[d] += it.V
                total  += it.V
                if vols[d] > max:
                    max = d
    numEmpty = 0
    for p in pallets:
        if p.Dest[k] == -1:
            numEmpty += 1

    for n in nodes:
        if vols[n.ID] > 0:
            np = math.floor( numEmpty * vols[n.ID] / total)
            count = 0
            for p in pallets:
                if count > np:
                    break
                if p.Dest[k] == -1:
                    pallets[p.ID].Dest[k] = n.ID
                    count += 1

    for p in pallets:
        if p.Dest[k] == -1:
            pallets[p.ID].Dest[k] = max
# end of setPalletsDestinations
 

def fillPallet(pallet, items, k, nodeTorque, solDict, cfg, vthreshold, itemsDict, lock, ts=1.0):
    N = len(items)
    counter = 0
    for item in items:
        if pallet.isFeasible(item, vthreshold, k, nodeTorque,cfg, itemsDict, lock, ts):
            pallet.putItem(item,nodeTorque,solDict,N, itemsDict, lock)
            counter += 1
    return counter

