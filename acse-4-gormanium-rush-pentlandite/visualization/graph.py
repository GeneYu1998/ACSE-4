"""@package docstring
File: This module is used to visualize the results of the output vector
"""

import sys
import graphviz

def loadDataset(file,k):
    """
    This function is used to load the previous output vector
    Parameters:
        file: A txt file with output vector saved
        k: 21 for most cases
    Return:
        A python vector nested anther python vector
    >>> loadDataset('output.txt', 21)
    [[3, 4, 5, 9, 3, 1, 11, 4, 7, 9, 8, 4, 6, 8, 2, 4, 0, 9, 1, 10, 4]]
    """
    f = open(file, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split(',')
        dataset.append(temp2)
    for i in range(0,len(dataset)):
        for j in range(k):
            dataset[i].append(int(dataset[i][j]))
        del(dataset[i][0:k])
    return dataset

def plot(output):
    """
    This function is to plot a flow chart.
    Parameters:
        Output: A python vector.
    Return：
        If the graph is plot succussfully, you can see 'Plot successfully!' in the terminal.
    >>> plot([3, 4, 5, 9, 3, 1, 11, 4, 7, 9, 8, 4, 6, 8, 2, 4, 0, 9, 1, 10, 4])
    Plot successfully!
    """
    # start with a graphviz object
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')
    graph.attr('node', shape='rectangle')

    # plotting
    initial = 'Unit ' + str(output[0])
    graph.edge('Feed', initial, color='black', headport='w', tailport='e', arrowhead='normal', arrowtail='normal')
    for i in range(max(output)-1):
        start = 'Unit ' + str(i)
        if output[2*(i+1)-1] == max(output)-1:
            end1 = 'Concentrate'
        elif output[2*(i+1)-1] == max(output):
            end1 = 'Tailings'
        else:
            end1 = 'Unit ' + str(output[2*(i+1)-1])
        if output[2*(i+1)] == max(output)-1:
            end1 = 'Concentrate'
        elif output[2*(i+1)] == max(output):
            end2 = 'Tailings'
        else:
            end2 = 'Unit ' + str(output[2*(i+1)])
    graph.edge(start, end1, color='blue', headport='w', tailport='e', arrowhead='normal', arrowtail='normal')
    graph.edge(start, end2, color='blue', headport='w', tailport='e', arrowhead='normal', arrowtail='normal')

    graph.node('Concentrate', shape='Mdiamond')
    graph.node('Tailings', shape='Mdiamond')

    # write png
    graph.render(filename='example', cleanup=True, format='png')

    return print("Plot successfully!")


# reaad in the vector and plot it
output1 = 'output.txt'
k = 21
output1 = loadDataset(output1, k)
final = [y for x in output1 for y in x]
print('Dataset =', final)
plot(final)


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)
