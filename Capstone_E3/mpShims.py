import numpy as np
import multiprocessing as mp
import time
import math
import os
import copy
import common
import optcgcons


# Este código implementa un enfoque de optimización para cargar ítems en pallets de manera que se minimicen las desviaciones del 
# centro de gravedad y se optimice la carga en términos de peso y volumen. A continuación, te doy un resumen de lo que hace cada 
# parte del código:



class Shims(object): # cabe en un pallet
    def __init__(self, pallet, whipLen):
        self.Pallet = pallet # this shims is one of the possibles for this pallet
        self.SCW    = 0      # shims current weight (must fit the slack)
        self.SCV    = 0.0    # shims current volume (must fit the slack)
        self.SCS    = 0      # shims current score
        self.SCT    = 0.     # shims current torque
        self.Items  = [None for _ in np.arange(whipLen)]

    # Agrega un ítem a la shim, actualizando su peso, volumen, puntaje y torque.

    def putItem(self, item, w):
                 
            self.SCW += item.W
            self.SCV += item.V
            self.SCS += item.S
            self.SCT += float(item.W) * float(self.Pallet.D)
            self.Items[w] = item

    # Verifica si un ítem es factible para agregar al pallet en función de su peso, volumen, destino, y torque.

    def isFeasible(self, item, k, nodeTorque, cfg, lock): # check constraints

        if item.To != self.Pallet.Dest[k]:
            return False

        if self.Pallet.PCV + self.SCV + item.V > self.Pallet.V:
            return False

        if self.Pallet.PCW + self.SCW + item.W > self.Pallet.W:
            return False

        deltaTau = float(item.W) * float(self.Pallet.D)
        ret = True
        with lock:
            newTorque = nodeTorque.value + self.SCT + deltaTau

            if  cfg.maxTorque < abs(newTorque):
                ret = False

        return ret

# create a set of shims for this pallet and selects the best shims
def getBestShims(pallet, items, k, nodeTorque, solDict, cfg, eta2_vol, itemsDict, lock, tipo, startTime, secBreak):
    # Crea y selecciona la mejor shim (o conjunto de shims) para un pallet dado.

    vol = 0.
    whip = []
    N = len(items)
    i = pallet.ID
   
   # 1. create the whip

   # Utiliza un bloqueo (with lock) para sincronizar el acceso compartido a datos, para un entorno concurrente.
   # Para cada ítem en items, si itemsDict["mpItems"][j] es 0 (probablemente indica que el ítem aún no se ha utilizado o asignado), 
   # se suma su volumen (item.V) al volumen acumulado (vol), 
   # y se añade el ítem a whip. Si el volumen acumulado (vol) supera el límite máximo del pallet (pallet.V), ajustado por el factor eta2_vol, 
   # el bucle se interrumpe. Esto asegura que el pallet no se sobrecargue en cuanto a volumen.

   # La función selecciona una serie de ítems que aún no han sido asignados y que juntos no superen el volumen permitido del pallet.

    with lock:   
        for item in items:
            j = item.ID
            if itemsDict["mpItems"][j] == 0:
                vol += item.V
                whip.append(item)
                if vol > pallet.V * eta2_vol:
                    break

    # 2. create the first shim
    # Crea un nuevo objeto Shims, representando un conjunto de ítems para el pallet.
    # El shim es inicializado con el pallet y el número de ítems en whip.
    # Finalmente, este shim se almacena en una lista llamada Set

    #  crea un "shim" (conjunto de ítems) inicial basado en esta selección y lo almacena en una lista llamada Set
    # para posteriormente ser usada para optimizar la distribución de ítems en el pallet.

    sh = Shims(pallet, len(whip))
    Set = [sh]

    if tipo == "FFD":
        # First Fit Decrease - faster than KP
        whip.sort(key=lambda x: abs(x.V), reverse=True)

        for w, item in enumerate(whip):

            if ((time.perf_counter() - startTime) > secBreak):
                break 

            newShims = True
            for sh in Set:
                if sh.isFeasible(item, k, nodeTorque, cfg, lock):
                    sh.putItem(item, w)
                    newShims = False
                    break

            if newShims:
                sh = Shims(pallet, len(whip)) # create a new Shim
                if sh.isFeasible(item, k, nodeTorque, cfg, lock):
                    sh.putItem(item, w)
                Set.append(sh)

    if tipo == "BF": ##3 ESTA VAMOS A LEGIR

        # Para cada ítem en whip (la lista de ítems preseleccionados que aún no están asignados a ningún shim), 

        for w, item in enumerate(whip): # for each item

            # el bucle evalúa si se ha alcanzado el tiempo máximo permitido (secBreak) desde que inició la función. 
            # Si el tiempo límite se alcanza, se interrumpe el bucle.

            if ((time.perf_counter() - startTime) > secBreak):
                break 

            # calculate the shims slacks

            # best_fit se inicializa con un valor alto (inf), representando la menor 
            # diferencia posible entre el espacio restante de un shim y el volumen del 
            best_fit = float('inf')


            # ítem.best_fit_ix se inicia en -1, que será el índice del shim que mejor ajuste tenga para el ítem.
            best_fit_ix = -1

            # Calcula el "slack" o espacio restante en el shim (sh_slack) restando el volumen actual del shim (sh.SCV) 
            # del volumen total del pallet. 
            # Si el slack menos el volumen del ítem es menor que best_fit, actualiza best_fit y best_fit_ix para 
            # reflejar este shim como el mejor ajuste.

            for s, sh in enumerate(Set):

                sh_slack = pallet.V - sh.SCV

                if sh_slack - item.V < best_fit:
                    best_fit = sh_slack - item.V
                    best_fit_ix = s # index of the shims that best fits the item
            
            # Si best_fit_ix es válido, significa que hay un shim en Set que tiene espacio para el ítem. Este ítem se agrega a ese shim.
            if best_fit_ix > -1: # fit the item or ...
                sh = Set[best_fit_ix]
                sh.putItem(item, w)
            else:
                # Si no hay un shim adecuado, crea uno nuevo (Shims(pallet, len(whip))), le agrega el ítem y luego lo añade a Set.
                sh = Shims(pallet, len(whip)) # create a new Shim
                sh.putItem(item, w)
                Set.append(sh)
    
    # select the best Shim
    bestScore = 0
    bestIndex = 0

    # Se recorren todos los shims en Set para identificar aquel con el mejor puntaje (SCS).
    # Se selecciona el shim con el puntaje más alto (bestScore) y se guarda su índice en bestIndex.
    for i, shims in enumerate(Set):
        if shims.SCS > bestScore:
            bestScore = shims.SCS
            bestIndex = i
    # put the best Shim in the solution

    # Verifica si el ítem cumple con la factibilidad (pallet.isFeasible) para el pallet.
    # Si es factible, el ítem se coloca en el pallet (pallet.putItem), y se actualizan 
    # las estructuras nodeTorque, solDict, y itemsDict según corresponda.

    for item in Set[bestIndex].Items:
        if item != None and pallet.isFeasible(item, 1.0, k, nodeTorque,  cfg, itemsDict, lock):
            pallet.putItem(item, nodeTorque, solDict, N, itemsDict, lock)

#                                    eta 1    eta 2
def Solve(pallets, items, cfg, pi, k, eta1_vol, eta2_vol, secBreak, mode, nodeTorque, solDict, itemsDict, tipo):

    startTime = time.perf_counter()

    ts = 1.0

    if mode == "p":
        mode = "Parallel"
        ts = 2.0
    else:
        mode = "Serial"

    # lock: Bloqueo multiproceso para sincronización.
    # N y M: Representan la cantidad de ítems (items) y pallets (pallets), respectivamente.

    lock  = mp.Lock()
    # counter = 0

    N = len(items)
    M = len(pallets)
    
    print(f"\nShims heuristic for ACLP+RPDP ({pi}-{k})")        
    print(f"{N} items  {M} pallets")

    # nodeTorque2 = mp.Value('d', nodeTorque.value)
    # pallets2    = common.copyPallets(pallets)
    # solDict2 = dict(solDict)
    # itemsDict2  = dict(itemsDict)


    for i, _ in enumerate(pallets):
        common.fillPallet( pallets[i], items, k, nodeTorque, solDict, cfg, 1.0, itemsDict, lock, ts)

    if mode == "Parallel":

        # --- did not work as expected
        # optcgcons.minCGdev(pallets, k, nodeTorque, cfg)

        procs = [None for _ in pallets] # each pallets has its own process

        # parallel shims phase

        # Crea un proceso independiente para cada pallet usando getBestShims como función objetivo.
        # Cada proceso ejecuta la asignación de shims en paralelo, optimizando la ocupación de ítems.
        # Usa time.sleep(0.001) para evitar conflictos en el inicio simultáneo de procesos.

        for i, p in enumerate(pallets):
            procs[i] = mp.Process( target=getBestShims, args=( pallets[i], items, k,\
                 nodeTorque, solDict, cfg, eta2_vol, itemsDict, lock, tipo, startTime, secBreak) )
            time.sleep(0.001)                 
            procs[i].start()
        for p in procs:
            p.join()

    else: # serial: Ejecuta getBestShims secuencialmente para cada pallet.
        for i, _ in enumerate(pallets):            
            # get the best Shims for the pallet
            getBestShims( pallets[i], items, k, nodeTorque, solDict, cfg, eta2_vol, itemsDict, lock, tipo, startTime, secBreak)

    # try to complete the pallet   
    common.fillPallet( pallets[i], items, k, nodeTorque, solDict, cfg, 1.0, itemsDict, lock)

    # Tras asignar los ítems a los shims, intenta llenar completamente el pallet restante llamando nuevamente a common.fillPallet.
    # En esta última fase se usa eta1_vol = 1.0, asegurando que el pallet esté completamente lleno, si es posible, con ítems adicionales.





if __name__ == "__main__":

    print("----- Please execute module main -----")