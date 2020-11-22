import math
import xml.etree.ElementTree as ET
import pickle
import re
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate
import numpy as np
from skimage import draw
f = 'IAMonDo-db-1.0/078.inkml'

"""
    -   Interpolation required to imcrease image resolution
    -   Textblocks not getting plotted
    -   Figure out how to store annotations and also plot bounding boxes to
        check if the coordinates are correct
    -   Convert force values into image intensity values
"""

scale_factor = 100

def get_all_traces(doc_namespace, root):
    traces_all = []
    for trace_tag in root.findall('trace'):
        ID = trace_tag.get(doc_namespace + 'id')
        coords = []
        for coord in (trace_tag.text).replace('\n', '').split(','):
            coords.append(coord.strip())
        traces_all.append({'id': ID, 'coords':coords})
    return traces_all


def get_numbers(string):
    lst = re.findall('-?\d*\.*\d+', string)
    return [float(num) for num in lst]

def get_tree_root(f):
    tree = ET.parse(f)
    root = tree.getroot()
    return root
    
def parse_file(f):
    root = get_tree_root(f)
    doc_namespace = "{http://www.w3.org/XML/1998/namespace}"
    traces_all = get_all_traces(doc_namespace, root)

    """
        Loop for extracting the initial coords, velocity and acceleration values
    """

    for i in range(len(traces_all)):
        if len(traces_all[i]['coords'])>=3:
            traces_all[i] = {
                'id' : traces_all[i]['id'], 
                'x_initial' : get_numbers(traces_all[i]['coords'][0]), 
                'v_initial' : get_numbers(traces_all[i]['coords'][1]),
                'a_initial' : get_numbers(traces_all[i]['coords'][2]),
                'coords' : [get_numbers(string) for string in traces_all[i]['coords'][3:]]
            }
        elif len(traces_all[i]['coords']) == 2:
            traces_all[i] = { 
                'id' : traces_all[i]['id'], 
                'x_initial' : get_numbers(traces_all[i]['coords'][0]), 
                'v_initial' : get_numbers(traces_all[i]['coords'][1]),
                'a_initial' : [0, 0, 0, 0],
                'coords' : []
            }
        elif len(traces_all[i]['coords']) == 1:
            traces_all[i] = { 
                'id' : traces_all[i]['id'], 
                'x_initial' : get_numbers(traces_all[i]['coords'][0]), 
                'v_initial' : [0, 0, 0, 0],
                'a_initial' : [0, 0, 0, 0],
                'coords' : []
            }
    return traces_all

def get_coords(trace):
    coords = []
    coords.append(trace['x_initial'])
    x = trace['x_initial']
    v = trace['v_initial'] 
    a = trace['a_initial']
    if trace['coords']:
        for i in range(len(trace['coords'])):
            x = [x[j]+v[j] for j in range(4)]
            coords.append(x)
            v = [v[j] + a[j] for j in range(4)]
            a = trace['coords'][i] 
    else:
        flag1 = False
        flag2 = False
        for i in range(4):
            if v[i]!=0:
                flag1 = True
            if a[i]!=0:
                flag2 = True
        if flag1:
            coords.append([x[i] + v[i] for i in range(4)])
        if flag2 and flag1:
            coords.append([x[i] + v[i] + a[i] for i in range(4)])
            
    return {'id' : trace['id'], 'coords' : coords}

def get_coords_all(traces_all):
    coords_all = []
    for trace in traces_all:
        coords_all.append(get_coords(trace))
    return coords_all

def get_bounding_box(traces, coords_all):
    min_x = 10e20
    max_x = -10e20
    min_y = 10e20
    max_y = -10e20
    for idx in traces:
        for coords in coords_all:
            if idx == coords['id']:
                for coord in coords['coords']:
                    if coord[0] < min_x:
                        min_x = coord[0]
                    if coord[0] > max_x:
                        max_x = coord[0]

                    if coord[1] < min_y:
                        min_y = coord[1]
                    if coord[1] > max_y:
                        max_y = coord[1]
    return min_x, max_x, min_y, max_y


def parse_correction(child, coords_all):
    traces = []
    annotation = child.findall('annotation')[1]
    for trace in child.findall('traceView'):
        traces.append(trace.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(traces, coords_all)
    return traces, annotation, box_vertices

def parse_word(child, coords_all):
    word = []
    text = ''
    try:
        text = child.findall('annotation')[1].text
        if text == None:
            text = ''
    except IndexError:
        text = ''
    for trace in child.findall('traceView'):
        if 'Correction' in [tag.text for tag in trace.findall('annotation')]:
            traces, annotation, box_vertices = parse_correction(trace, coords_all)
            word.append({
                'traces' : traces,
                'annotation' : annotation,
                'box' : box_vertices
            })
        else:
            #print([tag.text for tag in child.findall('annotation')])
            #text = child.findall('annotation')[1].text
            word.append(trace.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(word, coords_all)
    #print(word)
    return word, text, box_vertices

def parse_symbol(child, coords_all):
    symbol = []
    for trace in child.findall('traceView'):
        symbol.append(trace.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(symbol, coords_all)
    return symbol, box_vertices

def parse_textline(view, coords_all):
    textline = []
    sent = []
    for child in view.findall('traceView'):
        if child.findall('annotation')[0].text == 'Word':
            word, text, box_vertices = parse_word(child, coords_all)
            textline.append({
                'annotation' : 'Word',
                'word' : text,
                'traces' : word,
                'box' : box_vertices
            })
            #print(text)
            sent.append(text)
        elif child.findall('annotation')[0].text == 'Symbol':
            symbol, box_vertices = parse_symbol(child, coords_all)
            textline.append({
                'annotation' : 'Symbol',
                'traces' : symbol,
                'box' : box_vertices,
            })
    #print(sent)
    return textline, ' '.join(sent)

def parse_textblock(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        textline = []
        sent = ''
        if view.findall('annotation')[0].text == 'Textline':
            textline, sent = parse_textline(view, coords_all)
        traces.append({
            'annotation' : 'Textline',
            'text' : sent,
            'traces' : textline
        })
    return  traces

def parse_drawing(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        traces.append(view.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(traces, coords_all)
    return traces, box_vertices

def parse_diagram(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        textline = []
        if view.findall('annotation')[0].text == 'Textline':
            #print([tag.text for tag in view.findall('annotation')])
            textline, sent = parse_textline(view, coords_all)
            traces.append({
                'annotation' : 'Textline',
                'text' : sent,
                'traces' : textline
            })
        elif view.findall('annotation')[0].text == 'Drawing':
            drawing, box_vertices = parse_drawing(view, coords_all)
            traces.append({
                'annotation' : 'Drawing',
                'traces' : drawing,
                'box' : box_vertices
            })
        elif view.findall('annotation')[0].text == 'Structure':
            lst, structure, box_vertices = parse_structure(view, coords_all)
            traces.append({
                'annotation' : 'Structure',
                'structure' : structure,
                'traces' : lst,
                'box' : box_vertices
            })
        elif view.findall('annotation')[0].text == 'Arrow':
            lst, arrow, box_vertices = parse_structure(view, coords_all)
            traces.append({
                'annotation' : 'Structure',
                'structure' : arrow,
                'traces' : lst,
                'box' : box_vertices
            })
        elif view.findall('annotation')[0].text == 'Formula':
            formula, box_vertices = parse_formula(traceView, coords_all)
            traces.append({'annotation' : 'Formula', 'traces' : formula, 'box' : box_vertices})
    return  traces

def parse_structure(view, coords_all):
    lst = []
    structure = view.findall('annotation')[1].text
    for trace in view.findall('traceView'):
        lst.append(trace.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(lst, coords_all)
    return lst, structure, box_vertices

def parse_arrow(view, coords_all):
    lst = []
    arrow = view.findall('annotation')[1].text
    for trace in view.findall('traceView'):
        lst.append(trace.attrib['traceDataRef'][1:])
    box_vertices = get_bounding_box(lst, coords_all)
    return lst, arrow, box_vertices

def parse_formula(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        #print(view.attrib)
        #print(view.tag)
        #print([child for child in view.findall('traceView')])
        try:
            traces.append(view.attrib['traceDataRef'][1:])
        except KeyError:
            print('formula missing a trace')
    box_vertices = get_bounding_box(traces, coords_all)
    return traces, box_vertices

def parse_table(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        textline = []
        if view.findall('annotation')[0].text == 'Textline':
            textline, sent = parse_textline(view, coords_all)
            traces.append({
                'annotation' : 'Textline',
                'text' : sent,
                'traces' : textline
            })
        elif view.findall('annotation')[0].text == 'Structure':
            lst, structure, box_vertices = parse_structure(view, coords_all)
            traces.append({
                'annotation' : 'Structure',
                'structure' : structure,
                'traces' : lst,
                'box' : box_vertices,
            })
    return  traces 

def parse_list(traceView, coords_all):
    traces = []
    for view in traceView.findall('traceView'):
        textline = []
        sent = ''
        if view.findall('annotation')[0].text == 'Textline':
            textline, sent = parse_textline(view, coords_all)
        traces.append({
            'annotation' : 'Textline',
            'text' : sent,
            'traces' : textline
        })
    return  traces

def parse_annotations(root, coords_all):
    doc = []
    annotation = ''
    document = root.findall('traceView')[0]
    if document.findall('annotation')[0].text == 'Document':
        traceViews = document.findall('traceView')
        for traceView in traceViews:
            try:
                annotation = traceView.findall('annotation')[0].text
            except IndexError:
                annotation = ''
            if annotation == 'Textblock':
                traces = parse_textblock(traceView, coords_all)
                doc.append({'annotation' : 'TextBlock', 'traces' : traces})
            elif annotation == 'Drawing':
                traces, box_vertices = parse_drawing(traceView, coords_all)
                doc.append({'annotation' : 'Drawing', 'traces' : traces, 'box' : box_vertices})
            elif annotation == 'Diagram':
                traces = parse_diagram(traceView, coords_all)
                doc.append({'annotation' : 'Diagram', 'traces' : traces})
            elif annotation == 'List':
                traces = parse_list(traceView, coords_all)
                doc.append({'annotation' : 'List', 'traces' : traces})
            elif annotation == 'Table':
                traces = parse_diagram(traceView, coords_all)
                doc.append({'annotation' : 'Table', 'traces' : traces})
            elif annotation == 'Formula':
                traces, box_vertices = parse_formula(traceView, coords_all)
                doc.append({'annotation' : 'Formula', 'traces' : traces, 'box' : box_vertices})
            else:
                continue 
    return doc  

def _plot(trace, coords, axes):
    if type(trace) == str:
        for coord in coords:        
            if coord['id'] in trace:
                x = []
                y = []
                for i, values in enumerate(coord['coords']):
                    x.append(values[0])
                    y.append(values[1])
    else:
        for child in trace['traces']:
            _plot(child, coords, axes) 

def plot(doc, coords_all, name, folder):
    fig = plt.figure()
    axes = fig.gca()
    axes.invert_yaxis()
    axes.set_aspect('equal', adjustable='box')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    for child in doc:
        _plot(child, coords_all, axes)
    fig.savefig(os.path.join(folder, name[:-6]+'.png'), dpi = 100)
    fig.clear()
    plt.close(fig)
   
def save_annotation(doc, name, folder):
    pkl = open(os.path.join(folder, name[:-6]+'.pickle'), 'wb')
    pickle.dump(doc, pkl)
    pkl.close()
 
def transform_data(folder):
    data = os.listdir(folder)[880:]
    data = ['001.inkml']
    for f in tqdm(data):
        if 'set' in f:
            continue
        name = f
        f = os.path.join(folder, f)
        print(f)
        traces_all = parse_file(f)
        coords_all = get_coords_all(traces_all)
        root = get_tree_root(f)
        doc = parse_annotations(root, coords_all)
        if not os.path.exists('images'):
            os.mkdir('images')
        if not os.path.exists('annotations'):
            os.mkdir('annotations')
        plot(doc, coords_all, name, '.')
        save_annotation(doc, name, '.')

transform_data('IAMonDo-db-1.0')
