# import gurobipy as gp
# from gurobipy import GRB

from mip import Model, xsum, minimize, BINARY, CBC

import common

# optimize consolidated positions to minimize CG deviation
def OptCGCons(kept, pallets, k, nodeTorque):

    KeptRange = range(len(kept))
    PalletsRange = range(len(pallets))

    mod = Model(solver_name=CBC)
    mod.verbose = 0  # hide messages

    # Definir la variable binaria X para cada pallet y item
    X = [[mod.add_var(name=f"X[{i}],[{j}]", var_type=BINARY) for j in KeptRange] for i in PalletsRange]

    # Calcular torques a la izquierda (torque1) y derecha (torque2)
    torque1 = xsum(X[i][j] * ((140 + kept[j].W) * pallets[i].D) for i in PalletsRange for j in KeptRange if pallets[i].D > 0)
    torque2 = xsum(X[i][j] * ((140 + kept[j].W) * pallets[i].D) for i in PalletsRange for j in KeptRange if pallets[i].D < 0)

    # Definir la función objetivo (minimizar la diferencia de torques)
    mod.objective = minimize(torque1 - torque2)

    # Restricciones
    for j in KeptRange:
        mod.add_constr(xsum(X[i][j] for i in PalletsRange) == 1)
    for i in PalletsRange:
        mod.add_constr(xsum(X[i][j] for j in KeptRange) <= 1)

    # Resolver el modelo de optimización
    mod.optimize()

    # Inicializar los torques izquierdo y derecho
    torque_izquierda = 0
    torque_derecha = 0

    nodeTorque.value = 0
    # Iterar sobre los pallets para calcular los torques
    for i, _ in enumerate(pallets):
        pallets[i].Dest[k] = -1  # Resetear las destinaciones de los pallets desde este nodo
        nodeTorque.value += 140.0 * pallets[i].D
        for j in KeptRange:
            if X[i][j].x >= 0.99:
                nodeTorque.value += float(kept[j].W) * pallets[i].D
                pallets[i].Dest[k] = kept[j].To
                kept[j].P = i  # Colocar el item consolidado en la mejor posición para minimizar el torque

                # Calcular el torque según la distancia
                distancia = pallets[i].D  # Ajusta si es necesario
                if distancia < 0:  # A la izquierda del CG
                    torque_izquierda += kept[j].W * distancia
                else:  # A la derecha del CG
                    torque_derecha += kept[j].W * distancia


    # Imprimir el torque total a la izquierda y derecha del CG
    print(f"Torque total izquierdo: {torque_izquierda}")
    print(f"Torque total derecho: {torque_derecha}")
                 



if __name__ == "__main__":

    print("----- Please execute module main -----")
   