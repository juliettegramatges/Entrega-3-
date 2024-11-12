import common
import numpy as np
import time
import multiprocessing as mp
import os
import math
import pandas as pd
import csv

import mpShims
import optcgcons

from py3Druiz import Packer, Item, Bin

def solveTour(fecha_fin, fecha_inicio, vuelo, scenario, instance, pi, tour, method, pallets, cfg, tourTime, tipo, numOptDict,
               rampDistCG, afterDict, eta1_vol, eta2_vol, beforeDict, paquetes_df, aeropuertos):
    """
    Solves one tour
    """
    writeConsFile = False
    # writeConsFile = True  # write text files, 1 for packed each packed items

    # print(f"----- Tour {pi},", end='')
    # for node in tour.nodes:
    #     print(f" {node.ICAO}", end='')
    # print()

    # a matrix for all consolidated in the tour: conjunto de items que se pueden consolidar en cada pallet
    #  item tomará valor  -1 si se queda como item individual, -2 if a consollidated (packed), parte siendo un packed vacío
    consol = [
                [ common.Item(-1, -2, 0, 0, 0, 0, 0., -1, -1) # an empty consolidated
                for _ in tour.nodes ]
                for _ in pallets # tour consolidated for each pallet
            ]
    
    # imprimo la matriz consol visualmente

    print("Conjunto de items (Packed) vacíos en cada pallet creados para cada nodo")
    print(f"Tour nodes: {tour.nodes}")
    print(f"Pallets: {pallets}")
    df_consol = pd.DataFrame(consol)
    print(df_consol)

    k0 = len(tour.nodes)-1 # the base index on return

    # a first tour iteration to calculate the total volume to be tested for inclusion the tour nodes.

    # Primera Iteración: Cálculo del Volumen Total del Tour
    # En esta primera iteración, el código calcula el volumen total (tourVol) del tour y va pasando por cada nodo en tour.nodes:
    tourVol = 0.0
    for k, node in enumerate(tour.nodes):  # solve each node sequentialy
        print(f"Node {node.ICAO}")

        # L_k destination nodes set
        unattended = [n.ID for n in tour.nodes[k+1:]]

        # node max volume is updated
        _, node = common.loadNodeItems(scenario, instance, node, unattended, paquetes_df, aeropuertos)

        tourVol += node.Vol

    # a second tour iteration solving node-by-node
    # Segunda Iteración: Resolución Nodo por Nodo
    # Después de calcular tourVol, se realiza una segunda iteración para resolver el estado de cada nodo:

    for k, node in enumerate(tour.nodes):  # solve each node sequentialy

        # the solving node time is proportional to its relative sum of candidate items volumes
        if tourVol != 0:
            node.tLim = (node.Vol / tourVol) * tourTime
        else:
            # Manejo alternativo si tourVol es cero, por ejemplo, asignando un valor predeterminado
            node.tLim = 0  # O el valor adecuado en caso de que el volumen del tour sea cero


        # VEO EL SIGUIENTE NODO

        next = tour.nodes[k0]
        if k < k0:
            next = tour.nodes[k+1]

        # PARTO CON TORQUE 0

        nodeTorque = mp.Value('d', 0.0) # a multiprocessing double type variable
        nodeVol    = 0.0

        # LE AGREGO EL TORQUE DE LAS PALLETS INICIALES VACÍAS
        
        for i, p in enumerate(pallets):
            pallets[i].reset(cfg.numNodes) # resets pallets current weight (PCW) as 140kg
            nodeTorque.value += p.D * p.PCW # empty pallet torque

        # initialize- the accumulated values
        if writeConsFile:  # write text files, 1 for packed each packed items
            wNodeAccum = 0.
            vNodeAccum = 0.


        # BUSCO LOS DESTINOS NO VISITADOS
        # L_k destination nodes set
        unattended = [n.ID for n in tour.nodes[k+1:]]


        # CARGO LOS ITEMS QUE HAY EN UN NODO (NO LOS SUBO AÚN)
        # load items parameters from this node and problem instance, that go to unnatended
        items, _ = common.loadNodeItems(scenario, instance, node, unattended, paquetes_df, aeropuertos)


        # EMPIEZO A VISITAR LOS NODOS
        if k > 0 and k < k0: # no ha llegado al final

            # Utiliza la lista consol para cargar el contenido consolidado (CONJUNTO DE ITEMS) de los pallets en el nodo anterior (k-1). 
            # La lista cons recopila estos consolidados y les asigna ID temporales si se van a mantener en el tour.

            # Este fragmento crea una lista cons que contiene todos los ítems consolidados en los pallets del nodo anterior al actual. 
            # Esto permite que estos ítems se evalúen y procesen en el nodo actual, de modo que puedan conservarse en el tour si su destino 
            # aún no ha sido alcanzado.

            cons = []
            for i, _ in enumerate(pallets): # previous node consolidated
                    cons.append( consol[i][k-1] )

            # Para cada elemento en cons, verifica si su destino está en los nodos no atendidos (unattended). 
            # Si es así, el ítem se mantiene en el tour, se le asigna un nuevo ID y se agrega a kept para que 
            # se siga considerando en los próximos nodos.

            kept = []
            cid = N
            for c in cons:
                if c.To in unattended:
                    c.ID = cid
                    kept.append(c) #... and included in the items set
                    cid += 1

            # Optimize consolidated positions to minimize CG deviation.
            # Pallets destinations are also set, according to kept on board in new positions
            # Kept P is not -2 anymore, but the pallet ID.

            # Si kept contiene ítems, se llama a la función optcgcons.OptCGCons() para optimizar 
            # la posición de los ítems en el pallet. Esto minimiza la desviación del centro de gravedad 
            # para mantener estabilidad.

            if len(kept) > 0:

                optcgcons.OptCGCons(kept, pallets, k, nodeTorque)

                # N: number of items to embark
                for c in kept:
                    for i, p in enumerate(pallets):
                        if c.P == p.ID:
                            pallets[i].putConsol(c)

                            # update the consolidated of the current node "k"
                            consol[i][k].ID  = j+N
                            consol[i][k].Frm = node.ID
                            consol[i][k].To  = pallets[i].Dest[k]
                            consol[i][k].W  += c.W
                            consol[i][k].V  += c.V
                            consol[i][k].S  += c.S

                            # update the accumulated values
                            if writeConsFile:  # write text files, 1 for packed each packed items
                                wNodeAccum += c.W
                                vNodeAccum += c.V

                            tour.score += c.S
                            nodeVol    += c.V

        common.setPalletsDestinations(items, pallets, tour.nodes, k, unattended)

        startNodeTime = time.perf_counter()

        # COMIENZO A SUBIR ITEMS DE LOS NODOS AL AVIÓN

        # to control solution items
        M = len(pallets)
        N = len(items)

        if N > 0 and M > 0:

            print(f"Node {node.ICAO} {len(items)} items {len(pallets)} pallets")
            # mp.Array to be shared by multiprocess jobs
            solMatrix = mp.Array('i', [0 for _ in np.arange(N*M)] )
            mpItems   = mp.Array('i', [0 for _ in np.arange(N)] ) # to check items inclusions feasibility

            # I use dict to pass by reference
            solDict   = dict(solMatrix=solMatrix)
            itemsDict = dict(mpItems=mpItems)
            modStatus = 0

            # nodeTorque -> TORQUE PALLET VACÍO
            if method == "mpShims":
                # SE LLENAN LAS PALLETS CON LA MEJOR COMBINACIÇON DE SHIMS (CONJUNTO DE ITEMS) E ITEMS INDIVIDUALES
                mpShims.Solve(pallets, items, cfg, pi, k, eta1_vol, eta2_vol, node.tLim, "p", nodeTorque, solDict, itemsDict, tipo) # p - parallel

            if modStatus == 2: # 2: optimal
                numOptDict["numOpt"] += 1


            nodeElapsed = time.perf_counter() - startNodeTime

            Y = np.reshape(solDict["solMatrix"], (-1, N)) # N number of items (columns)
            # La matriz de soluciones (solMatrix) almacenada en solDict se reorganiza en una forma que tiene tantas filas 
            # como pallets y tantas columnas como ítems. N es el número de ítems. Esta matriz representa qué ítems están 
            # asignados a qué pallets.
        
            # begin ---- parallel solving the 3D packing for each pallet 

            # Se crean listas para almacenar los procesos y los empaquetadores (Packer).
            # Para cada fila de la matriz Y, que representa un pallet, se crea un nuevo empaquetador (Packer).
            # Se agrega un Bin (contenedor) para cada pallet, con sus dimensiones y peso.
            # Si el valor en la matriz Y es 1 (lo que indica que un ítem está asignado a este pallet), se agrega el ítem correspondiente al Packer.
            # Luego, se inicia un proceso paralelo (mp.Process) para ejecutar la función de empaquetado del Packer en paralelo.
            try:
                procs   = [None for _ in pallets]
                packers = [None for _ in pallets]
                counter1 = 0
                for i, row in enumerate(Y):
                    packers[i] = Packer()
                    packers[i].add_bin( Bin(f'pallet{i}', pallets[i].w, pallets[i].h, pallets[i].d, pallets[i].W, i) )

                    for j, X_ij in enumerate(row):
                        if X_ij:
                            packers[i].add_item(Item(f'item{j}', items[j].w, items[j].h, items[j].d, 0, j))
                            counter1 += 1
                            print(f"Item {j} (ID: {items[j].ID}) asignado al pallet {i + 1}")  # Imprimir ítem asignado
                    procs[i] = mp.Process(target=packers[i].pack())
                    procs[i].start()

                counter2 = 0
                for i, proc in enumerate(procs):
                    proc.join()
                    for bin in packers[i].bins:
                        i = bin.ID
                        for item in bin.unfitted_items:
                            j = item.ID
                            Y[i][j] = 0
                            counter2 += 1
                            pallets[i].popItem(items[j], nodeTorque, solDict, N, itemsDict)
                            print(f"Item {j} (ID: {items[j].ID}) no ajustado en pallet {i + 1}, se baja")  # Imprimir ítem no ajustado

                if counter1 > 0:
                    print(f"{100 * counter2 / counter1:.1f}% unfit items excluded from solution!")
                    
            except Exception as e:
                print("Error en 3D")
                print(e)
                pass

            nodeElapsed2 = time.perf_counter() - startNodeTime
            # end ---- parallel solving the 3D packing for each pallet         

            # Se inicializan variables para el puntaje del nodo (nodeScore) y el torque acumulado (torque).

            nodeScore = 0
            torque    = 0.0

            for i, row in enumerate(Y):

                torque += 140 * pallets[i].D

                for j, X_ij in enumerate(row):
                    if X_ij:
                        # mount this node "k" consolidated
                        consol[i][k].ID  = j+N
                        consol[i][k].Frm = node.ID
                        consol[i][k].To  = pallets[i].Dest[k]                    
                        consol[i][k].W += items[j].W
                        consol[i][k].V += items[j].V
                        consol[i][k].S += items[j].S

                        # Se recorre la matriz Y, que contiene la asignación de los ítems a los pallets. 
                        # Para cada ítem asignado (X_ij es verdadero), se actualizan varias propiedades del nodo consolidado 
                        # (consol[i][k]), como el ID del ítem, el peso (W), el volumen (V) y la superficie (S).
                        # Se acumulan también los parámetros relacionados con el puntaje del nodo (nodeScore) y el volumen (nodeVol).

                        # totalize parameters of this solution
                        if writeConsFile: # write text files, 1 for packed each packed items
                            wNodeAccum += float(items[j].W)
                            vNodeAccum += float(items[j].V)

                        nodeScore  += items[j].S
                        nodeVol    += items[j].V

                        torque += float(items[j].W) * pallets[i].D

                        # Se acumula el puntaje total (BENEFICIO TOTAL) del nodo con el valor de la superficie de cada ítem (items[j].S), 
                        # el volumen total del nodo (nodeVol) y el torque generado por cada ítem.

                        # Imprimir la distribución de los ítems en los pallets
                        print(f"Distribución de items en los pallets una vez llegó al nodo {node.ICAO}:")

                        # Lista para llevar un registro de los paquetes que siguen a bordo y los que se bajan
                        paquetes_a_bordo = []
                        paquetes_bajados = []

                        # Recorremos cada pallet y sus asignaciones de items
                        for i, row in enumerate(Y):
                            print(f"Pallet {i + 1}:")
                            for j, X_ij in enumerate(row):
                                if X_ij:
                                    # Imprimir el paquete asignado al pallet
                                    print(f"  Item {j + 1} (ID: {items[j].ID}) asignado al pallet {i + 1}")
                                    print(f"  Destino del paquete: {items[j].To}")

                                    paquetes_df.loc[paquetes_df['id'] == (items[j].ID - 1), 'assigned'] = True

                                    # El paquete sigue a bordo
                                    if X_ij == 1:
                                        paquetes_a_bordo.append(items[j].ID)
                                        print(f"    Paquete {items[j].ID} sigue a bordo en el nodo {node.ICAO}")
                                    # El paquete se ha bajado
                                    elif X_ij == 0:
                                        paquetes_bajados.append(items[j].ID)
                                        print(f"    Paquete {items[j].ID} se ha bajado en el nodo {node.ICAO}")

                        # Imprimir los paquetes que siguen a bordo
                        print(f"\nPaquetes que siguen a bordo después de este nodo {node.ICAO}:")
                        for paquete in paquetes_a_bordo:
                            print(f"  Paquete {paquete}")

                        # Imprimir los paquetes que se han bajado
                        print(f"\nPaquetes que se han bajado en este nodo {node.ICAO}:")
                        for paquete in paquetes_bajados:
                            print(f"  Paquete {paquete}")

   

            output_folder = os.path.join("resultados", f"{fecha_inicio}-{fecha_fin}" , f"{scenario}")
            os.makedirs(output_folder, exist_ok=True)

            # Definir el archivo CSV de salida dentro de la carpeta `resultados/vueloid`
            output_csv_path = os.path.join(output_folder, "resultados.csv")

            # Columnas del archivo CSV
            header = ["Paquete_ID", "Pallet_ID", "Nodo_ID", "Peso", "Volumen", "Superficie", "Torque"]

            # Abrir el archivo CSV en modo escritura
            with open(output_csv_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)  # Escribir la cabecera

                # Procesar cada pallet y nodo
                for i, row in enumerate(Y):
                    torque += 140 * pallets[i].D  # Torque inicial del pallet

                    for j, X_ij in enumerate(row):
                        if X_ij:
                            # Obtener datos del paquete y nodo
                            paquete_id = j + N
                            pallet_id = i
                            nodo_id = node.ID
                            peso = items[j].W
                            volumen = items[j].V
                            beneficio = items[j].S
                            torque_item = float(items[j].W) * pallets[i].D
                            # Escribir la fila con los datos en el archivo CSV
                            writer.writerow([paquete_id, pallet_id, nodo_id, peso, volumen, torque_item])

    # Se actualizan los valores de tiempo transcurrido en el tour (tour.elapsed, tour.elapsed2).
            tour.elapsed += nodeElapsed
            tour.elapsed2 += nodeElapsed2
    # El volumen del nodo se normaliza en relación con la capacidad de volumen (cfg.volCap).
            nodeVol /= cfg.volCap
    # Se calcula epsilon, que es la relación entre el torque acumulado y el torque máximo permitido (cfg.maxTorque).
            epsilon = torque/cfg.maxTorque
    # Si el volumen es mayor que 1.0, el puntaje del nodo se ajusta dividiéndolo por el volumen. Esto penaliza las soluciones con alto volumen.
            if nodeVol > 1.0:
                nodeScore /= nodeVol
    # Se actualizan las estadísticas del tour, incluyendo el puntaje total (tour.score), el volumen promedio (tour.AvgVol) y el torque promedio (tour.AvgTorque).
            tour.score += nodeScore
            tour.AvgVol    += nodeVol
            tour.AvgTorque += epsilon
            f = tour.score
        
            print(f"\tnode {node.ICAO}")
            #print(f"f {f:.2f}  vol {nodeVol:.2f} epsilon {epsilon:.2f}")


            if writeConsFile: # write text files, 1 for packed each packed items

                consNodeT = [None for _ in pallets]        
                for i, p in enumerate(pallets):
                    consNodeT[i] = consol[i][k]

                vol = vNodeAccum/cfg.volCap
                wei = wNodeAccum/cfg.weiCap

                #write consolidated contents from this node in file
                common.writeNodeCons(scenario, instance, consNodeT, pi, node, epsilon, wei, vol)


# end of solveTour 


def writeAvgResults(method, scenario, line, folder):

    dirname = f"./results/{folder}/"
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass  

    fname = f"{dirname}/{method}_{scenario}.avg"

    writer = open(fname,'w+') # + creates the file, if not exists

    try:
        writer.write(line)
    finally:
        writer.close()  

def writeResults(method, scenario, folder, fvalue, elapsed):

    dirname = f"./results/{folder}/"
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass  

    fname = f"{dirname}/{method}_{scenario}.res"    
        
    line = f"{elapsed},{fvalue}\n"

    writer = open(fname, "a+") 
    try:
        writer.write(line)
    finally:
        writer.close() 

TOTAL = 0
aviones = []

if __name__ == "__main__":


    # flexibilidad de volumen
    eta1_vol, eta2_vol = 1, 1

    mainStart = time.perf_counter()

    plot      = False

    testing   = True

    iRace_testing = False

    # shortest = True # 2 shortest tours
    shortest = False # All K! # esto podría ser TRUE AL trabajar con los cargo:


    df_vuelos = pd.read_csv('C:/Users/ignac/Downloads/Entrega-3--main/Entrega-3--main/Capstone_E3/flights_combined.csv')



    df_paquetes = pd.read_csv('C:/Users/ignac/Downloads/Entrega-3--main/Entrega-3--main/Capstone_E3/packages.csv')
    df_paquetes['assigned'] = False  # Initialize this before calling solveTour


    df_paquetes['due_date'] = pd.to_datetime(df_paquetes['due_date'])
    df_vuelos['departure_date'] = pd.to_datetime(df_vuelos['departure_date'])

    # Filtrar las filas dentro del rango de fechas (1 de enero - 14 de enero de 2024) AHORA PRUEBO ULTIMA QUINCENA DE ENERO
    fecha_inicio = '2024-01-15'
    fecha_fin = '2024-01-28'

    # Aplicar el filtro
    vuelos_filtrados = df_vuelos[(df_vuelos['departure_date'] >= fecha_inicio) & (df_vuelos['departure_date'] <= fecha_fin)]
    
    # Extraer los IDs de los vuelos filtrados en lugar de un rango numérico
    scenarios = vuelos_filtrados['id'].tolist() 

    print(len(scenarios))
    print(scenarios)

    if testing:
        scenarios = scenarios


    timeLimit = 240
    # timeLimit = 1200
    # timeLimit = 2400
    #timeLimit = 3600 # if there is any metaheuristics in the experiment (except Shims)
    method = "mpShims"

    #print(f"timeLimit:{timeLimit}   method: {method}   shortest: {shortest}")

    # PARA EL mpShims

    # tipo = "KP" # with dynamic programming          46.6    61
    # tipo = "FFD"  # with First-fit decreasing mainElapsed: 860.1    overall SC: 113.5
    tipo = "BF"  # with Best-fit        mainElapsed: 662.5    overall SC: 114.6

    ########### GUARDAMOS DISTANCIA ENTRE AEROPUERTOS ################

    overallSC = 0.0

    for scenario in scenarios:
        
        vuelo = df_vuelos.iloc[scenario-1]
        # Restar 2 días a la fecha
        paquetes_filtrados = df_paquetes[(df_paquetes['due_date'] >= (vuelo['departure_date'] - pd.Timedelta(days=2))) & 
                                 (df_paquetes['due_date'] <= vuelo['departure_date']) & 
                                 (~df_paquetes['assigned'])]

        #actualizo el beneficio de los paquetes de acuerdo a su penalización
        
        instances = [1]
        # tomo los nodos de la lista de destinos del escenario
        cfg = common.Config(vuelo, scenario)

        # guardo el nombre de los aeropuertos
        aeropuertos = {}
        for i in range(len(cfg.nodos)):
            aeropuertos[cfg.nodos[i]] = i
        
        distances_file = "./distances.csv"
        df = pd.read_csv('C:/Users/ignac/Downloads/Entrega-3--main/Entrega-3--main/Capstone_E3/distances.csv')

        # tomo solo las filas que tengan unicamente nodos del escenario
        distances_file_2 = df[df.iloc[:, 0].isin(cfg.nodos) & df.iloc[:, 1].isin(cfg.nodos)]
        #print(distances_file_2)

        dists, _ = common.loadDistances(distances_file_2) # dists, cities
        costs = [[0.0 for _ in dists] for _ in dists]

        # matriz costs de tamaño n x n, donde n es la cantidad de elementos en dists, y todos los valores son 0.0


        #print(f"timeLimit:{timeLimit}    method: {method}   shortest: {shortest}")
        #print(f"\n{cfg.numNodes} nodes")

        for i, cols in enumerate(dists):
            for j, dist in enumerate(cols):
                costs[i][j] = cfg.kmCost*dist
        
        df_costs = pd.DataFrame(costs, columns=cfg.nodos, index=cfg.nodos)
        #print(df_costs)
    
        # Este bloque de código recorre la matriz dists (distancias entre nodos) y calcula los costos entre nodos. 
        #  Para cada par de nodos i y j, el costo entre ellos se calcula multiplicando la distancia dist por un valor cfg.kmCost, 
        #  que representa un costo por kilómetro.

        pallets, rampDistCG = common.loadPallets(cfg)

        # pallets capacity
        cfg.weiCap = 0
        cfg.volCap = 0
        for p in pallets:
            cfg.weiCap += p.W
            cfg.volCap += p.V

        # smaller aircrafts may have a payload lower than pallets capacity
        if cfg.weiCap > cfg.payload:
            cfg.weiCap = cfg.payload

        perc = 1.0
        if cfg.numNodes > 3 and cfg.numNodes <= 7:
            perc = 0.25 # discard the worst tours

        numOptDict = {"numOpt":0}
        afterDict  = {"value":0.}
        beforeDict = {"value":0.}


        numInst = float(len(instances))

        avgInstTime       = 0
        instBestAvgVol    = 0.
        instBestAvgTorque = 0.        
        instBestAvgSC     = 0. # score/cost relation
        leastSC           = 0.


        if vuelo['Flight Type'] == 'Cargo':
            tours = common.getTours(cfg.numNodes, costs, perc, aeropuertos, "cargo")
        else:
            tours = common.getTours(cfg.numNodes, costs, perc, aeropuertos, "passenger")

        tourTime = timeLimit/len(tours)

        for inst in instances:

            now = time.perf_counter()

            instanceStartTime  = now

            bestSC = 0. # maximum score/cost relation
            bestAvgVol = 0.
            bestAvgTorque = 0.
            bestTourID = -1
            bestTour = []

            for pi, tour in enumerate(tours):

                tour.elapsed = 0
                tour.elapsed2 = 0 # with 3D packing
                tour.score   = 0.0
                tour.AvgVol  = 0.0

                

                solveTour(fecha_fin, fecha_inicio, vuelo, scenario, inst, pi, tour, method, pallets, cfg, tourTime, tipo, numOptDict, rampDistCG, afterDict, 
                            beforeDict, eta1_vol, eta2_vol, paquetes_filtrados, aeropuertos)
                

                
                for index, paquete in paquetes_filtrados.iterrows():
                    
                    # Cambiar el valor de 'assigned' en 'paquetes_filtrados' y reflejar en 'df_paquetes'
                    if paquete['assigned']: 
                        print("xd") 
                        df_paquetes.at[index-1, 'assigned'] = True
                    

                # the tour cost is increased by the average torque deviation, limited to 5%
                tour.AvgTorque /= cfg.numNodes 

                tourSC = tour.score 

                tour.AvgVol /= cfg.numNodes

                # best tour parameters
                if tourSC > bestSC:
                    bestSC        = tourSC
                    bestAvgVol    = tour.AvgVol
                    bestAvgTorque = tour.AvgTorque
                    bestTourID    = pi
                    bestTour      = tour.nodes
                   
            now = time.perf_counter()

            avgInstTime += now - instanceStartTime

            instBestAvgSC     += bestSC
            instBestAvgVol    += bestAvgVol
            instBestAvgTorque += bestAvgTorque        
            # enf of for inst in instances:
        
        ###### IMPRIME RESULTADOS ######

        folder = f"resultados/{scenario}"
            
        avgTime       = math.ceil(avgInstTime/numInst)
        bestAvgSC     = instBestAvgSC/numInst
        bestAvgVol    = instBestAvgVol/numInst
        bestAvgTorque = instBestAvgTorque/numInst

        aviones.append(scenario)
        TOTAL += bestAvgSC 
        

        icaos = []
        for n in bestTour:
            icaos.append(n.ICAO)

        # list the Shims best tour as an ICAO list
        origin = icaos[0]
        sbest_tour = f"{origin} "
        prev = icaos[0]
        for j, icao in enumerate(icaos):
            if j > 0:
                sbest_tour += f"{icao} "
                prev = icao

        print(f"Beneficio {bestAvgSC:.2f}\t&\t{avgTime:.0f}\t {bestAvgVol:.2f}\t {bestAvgTorque:.2f}")
        print(f"{len(tours)} tours")

        str = f"{bestAvgSC:.2f}\t&\t{avgTime:.0f}\t {bestAvgVol:.2f}\t {bestAvgTorque:.2f}"
        # instances average
        writeAvgResults(method, scenario, str, folder)
        

    if not iRace_testing:

        folder = "resultados_totales"

        print(f"Resultados avion nro: {scenario}")
        print(f"Beneficio total: {bestAvgSC:.2f}\t&\tTiempo promedio: {avgTime:.0f}\t Volumen promedio: {bestAvgVol:.2f}\t Torque promedio: {bestAvgTorque:.2f}")
        # print(f"{folder}")
        print(f"{len(tours)} tours")
        print(f"    best: {sbest_tour}")
        # print(f"shortest: {shortestTour}")

        mainElapsed = time.perf_counter() - mainStart

        overallSC += bestAvgSC

        print(f"mainElapsed: {mainElapsed:.1f}    overall SC: {overallSC:.3f}")

    else:
        print(-1*instBestAvgSC/numInst) # -1: iRace minimizes a cost value

#"""

print(aviones)
print(TOTAL)
