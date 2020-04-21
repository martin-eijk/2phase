# Packages
import xml.dom.minidom
import numpy as np
from pyevtk.hl import gridToVTK

""" 
For producing output for Paraview and save it 
"""

# Subfunctions
def makePVD():
    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd_root.setAttribute("compressor", "vtkZLibDataCompressor")
    pvd.appendChild(pvd_root)

    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)
    return pvd, collection
def makeVTM():
    vtm = xml.dom.minidom.Document()
    vtm_root = vtm.createElementNS("VTK", "VTKFile")
    vtm_root.setAttribute("type", "vtkMultiBlockDataSet")
    vtm_root.setAttribute("version", "1.0")
    vtm_root.setAttribute("byte_order", "LittleEndian")
    # vtm_root.setAttribute("header_type", "UInt64")
    vtm.appendChild(vtm_root)

    dataset = vtm.createElementNS("VTK", "vtkMultiBlockDataSet")
    vtm_root.appendChild(dataset)
    return vtm, dataset
def dataPVD(pvd, collection, time, VTK):
    for i in range(1, len(time) + 1):
        if i % VTK[0, 1] == 0:
            dataSet = pvd.createElementNS("VTK", "DataSet")
            dataSet.setAttribute("timestep", str(time[i-1]))
            dataSet.setAttribute("group", "")
            dataSet.setAttribute("part", "0")
            dataSet.setAttribute("file", './VTM/'+'Results'+str(time[i-1])+".vtm")
            collection.appendChild(dataSet)
    return pvd
def dataVTM(vtm, dataset, vtm_file, time):
    for xx in range(np.size(vtm_file, 0)):
        gg = np.array(vtm_file[xx])
        dataSet = vtm.createElementNS("VTK", "DataSet")
        dataSet.setAttribute("index", gg[1])
        dataSet.setAttribute("name", gg[0])
        dataSet.setAttribute("file", './Data/'+gg[0]+'/'+gg[0]+str(time)+".vts")
        dataset.appendChild(dataSet)
    return vtm
def writePVD(pvd, fileName):
    outFile = open(fileName+"/Results"+'.pvd', 'w')
    pvd.writexml(outFile, newl='\n')
    outFile.close()
def writeVTM(vtm, fileName, time):
    outFile = open(fileName+"/VTM/Results"+str(time)+'.vtm', 'w')
    vtm.writexml(outFile, newl='\n')
    outFile.close()
def writeVTKcell(fileName, data_name, data, time_level, spacingx, spacingy, spacingz):
    data2 = np.fliplr(np.flipud(np.rot90(data[2:np.size(spacingy, 2)+1, 2:np.size(spacingx, 0)+1])))
    plot_data = np.zeros((np.size(spacingx, 0) - 1, 1, np.size(spacingy, 2) - 1))
    plot_data[:, 0, :] = data2

    gridToVTK(fileName + str(time_level), spacingx, spacingz, spacingy, cellData={data_name: plot_data})

# Mainfunction
def VTKsave(VTK, VTKpath, spacingx, spacingy, spacingz, time, n, fraction, Uvel, Vvel, density, pressure, cell):
    if n % VTK[0, 1] == 0:
        vtm, dataset = makeVTM()
        pvd, collection = makePVD()
        vtm_file =[]
        if VTK[0, 2] == 1:
            writeVTKcell(VTKpath[1]+'/Fraction', "Fraction", fraction, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Fraction', 0])
        if VTK[0, 3] == 1:
            writeVTKcell(VTKpath[2]+'/Uvel', "Uvel", Uvel, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Uvel', 1])
            writeVTKcell(VTKpath[3]+'/Vvel', "Vvel", Vvel, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Vvel', 2])
        if VTK[0, 4] == 1:
            writeVTKcell(VTKpath[4]+'/Density', "Density", density, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Density', 3])
        if VTK[0, 5] == 1:
            writeVTKcell(VTKpath[5]+'/Pressure', "Pressure", pressure, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Pressure', 4])
        if VTK[0, 6] == 1:
            cell_plot = np.zeros([np.size(cell, 0), np.size(cell, 1)])
            for j in range(2, np.size(cell, 0)-2):
                for i in range(2, np.size(cell, 1)-2):
                    if cell[j, i] == 'F':
                        cell_plot[j, i] = 1
                    if cell[j, i] == 'E':
                        cell_plot[j, i] = 3
                    if cell[j, i] == 'S':
                        cell_plot[j, i] = 2
            writeVTKcell(VTKpath[6]+'/Cell', "Cell", cell_plot, time[n-1], spacingx, spacingy, spacingz)
            vtm_file.append(['Cell', 5])

        vtm = dataVTM(vtm, dataset, vtm_file, time[n-1])
        writeVTM(vtm, VTKpath[0], time[n-1])

        pvd = dataPVD(pvd, collection, time, VTK)
        writePVD(pvd, VTKpath[0])
