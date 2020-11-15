import xml.etree.ElementTree as ET
import re

f = 'IAMonDo-db-1.0/078.inkml'

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
    #print(traces_all[0])

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
    #print(trace)
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

def parse_textblock(traceView):
    return {}

def parse_drawing(traceView):
    traces = []
    for view in traceView.findall('traceView'):
        traces.append(view.attrib['traceDataRef'][1:])
    return {'traces' : traces}

def parse_diagram(traceView):
    return {}

def parse_table(traceView):
    return {}

def parse_list(traceView):
    return {}

def parse_annotations(root):
    doc = []
    document = root.findall('traceView')[0]
    if document.findall('annotation')[0] == 'Document':
        traceViews = document.findall('traceView')
        for traceView in traceViews:
            annotation = traceView.findall('annotation')[0]
            if annotation == 'TextBlock':
                labels = parse_textblock(traceView)
                doc.append({'annotation' : 'TextBlock', 'annotation' : labels})
            elif annotation == 'Drawing':
                labels = parse_drawing(traceView)
                doc.append({'annotation' : 'Drawing', 'annotation' : labels})
            elif annotation == 'Diagram':
                labels = parse_diagram(traceView)
                doc.append({'annotation' : 'Diagram', 'annotation' : labels})
            elif annotation == 'List':
                labels = parse_list(traceView)
                doc.append({'annotation' : 'List', 'annotation' : labels})
            elif annotation == 'Table':
                labels = parse_diagram(traceView)
                doc.append({'annotation' : 'Table', 'annotation' : labels})
            else:
                continue   

traces_all = parse_file(f)
coords_all = get_coords_all(traces_all)
