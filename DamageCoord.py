# ZX change / Read  :  2025/1/10
# -*- coding: utf-8 -*-
import odbAccess
from odbAccess import openOdb
from abaqusConstants import *
import visualization
#change the numbers in range(X,XX) based on odb names

# Set the base dir folder
base_dir = 'D:/PYTHON/Coord'

for jobn in range(1,2):
    name = os.path.join(base_dir, 'Artery{}.odb'.format(jobn))

    odbName = 'Artery%s.odb' % jobn
    myOdb = openOdb(odbName,readOnly=True)

    steps = list(myOdb.steps.values())

    # the 1st frame of the 1st Step
    first_step = steps[0]
    f1 = first_step.frames[0]
    # the last frame of the last Step
    last_step = steps[-1]
    f2 = last_step.frames[-1]


    myelement = myOdb.rootAssembly.elementSets['DAMAGE']

    damage=f2.fieldOutputs['SDV11'].getSubset(region=myelement)
    damagecoord=f2.fieldOutputs['COORD'].getSubset(region=myelement)
    coords = {}
    for coord in damagecoord.values:
        coords[coord.nodeLabel] = coord.dataDouble

    sumdamage = 0.0
    maxdamage = 0.0
    number= 0.0
    max_damage_coords = (0.0, 0.0, 0.0)
    # Initialize max damage coordinates

    for i in damage.values:
        damage_value = i.data
        sumdamage += damage_value
        number += 1
        if damage_value > maxdamage:
            maxdamage = damage_value

    len_damagecoord = len(coords)
    len_damage = len(damage.values)

    #for i,j in zip(damage.values,damagecoord.values):
        #damage_value = i.data
        #coord_value = j.data
        #sumdamage += damage_value
        #number += 1
        #if damage_value > maxdamage:
            #maxdamage = damage_value
            #max_damage_coords = coord_value

    # averagedamage = sumdamage / number
    if number != 0:  #if there are damage points, calculate the average damage
        averagedamage = sumdamage / number
    else:  # if not, acerage damage = 0
        averagedamage = 0

    #print(averagedamage)

    mynodes=myOdb.rootAssembly.nodeSets['PLAQUENODES']

    coordinate0 = f2.fieldOutputs['COORD'].getSubset(region=mynodes)
    coordinate1 = f1.fieldOutputs['COORD'].getSubset(region=mynodes)

    x0=[]
    y0=[]
    z0=[]
    x1=[]
    y1=[]
    z1=[]
    x0 = [vv.dataDouble[0] for vv in coordinate0.values]
    y0 = [vv.dataDouble[1] for vv in coordinate0.values]
    z0 = [vv.dataDouble[2] for vv in coordinate0.values]

    x1 = [ee.dataDouble[0] for ee in coordinate1.values]
    y1 = [ee.dataDouble[1] for ee in coordinate1.values]
    z1 = [ee.dataDouble[2] for ee in coordinate1.values]

    with open("artery%s.csv" % jobn,"w") as output:
     output.write(
      #  "%.7f,%.7f,%.7f,%.7f,%.7f\n" % 
        "%.7f,%.7f\n" % 
        (
            averagedamage,
            maxdamage
            #max_damage_coords[0], 
            #max_damage_coords[1], 
            #max_damage_coords[2]
        ))
     for j in range(len(x0)):
      output.write(
        "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f\n" %
        (
            x0[j], y0[j], z0[j], x1[j], y1[j], z1[j]
         ))
      # %.7f : Keep 7 decimal