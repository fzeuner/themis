"""
Generate a Graphviz documentation plot showing the THEMIS data reduction pipeline.

This script creates a visual representation of:
- Data flow from raw to L3
- Functions applied at each reduction level
- Data types (dark, scan, flat, flat_center)
- Auxiliary files generated

Run: python scripts/pipeline_documentation.py
Output: figures/pipeline_diagram.png (and .pdf)
"""

from graphviz import Digraph
from pathlib import Path

directory = '/home/franziskaz/themis/figures/'

def create_pipeline_diagram(output_dir=directory+'pipeline'):
    """Create a Graphviz diagram of the THEMIS data reduction pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create directed graph
    dot = Digraph('THEMIS_Pipeline', comment='THEMIS Data Reduction Pipeline')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # Color scheme
    colors = {
        'raw': '#FFE4B5',      # Moccasin (orange-ish)
        'l0': '#FFDAB9',       # Peach
        'l1': '#98FB98',       # Pale green
        'l2': '#87CEEB',       # Sky blue
        'l3': '#DDA0DD',       # Plum
        'auxiliary': '#F0E68C', # Khaki
        'function': '#E6E6FA', # Lavender
        'note': '#FFFFFF',     # White
    }
    
    # =========================================================================
    # DATA LEVELS (main nodes)
    # =========================================================================
    
    # Raw data
    with dot.subgraph(name='cluster_raw') as c:
        c.attr(label='RAW (.fts)', style='rounded', bgcolor=colors['raw'])
        c.node('raw_dark', 'dark\n(raw)', fillcolor=colors['raw'])
        c.node('raw_scan', 'scan\n(raw)', fillcolor=colors['raw'])
        c.node('raw_flat', 'flat\n(raw)', fillcolor=colors['raw'])
        c.node('raw_flat_center', 'flat_center\n(raw)', fillcolor=colors['raw'])
    
    # L0 data
    with dot.subgraph(name='cluster_l0') as c:
        c.attr(label='L0 (_l0.fits)', style='rounded', bgcolor=colors['l0'])
        c.node('l0_dark', 'dark\n(L0)', fillcolor=colors['l0'])
        c.node('l0_scan', 'scan\n(L0)', fillcolor=colors['l0'])
        c.node('l0_flat', 'flat\n(L0)', fillcolor=colors['l0'])
        c.node('l0_flat_center', 'flat_center\n(L0)', fillcolor=colors['l0'])
    
    # L1 data
    with dot.subgraph(name='cluster_l1') as c:
        c.attr(label='L1 (_l1.fits)', style='rounded', bgcolor=colors['l1'])
        c.node('l1_dark', 'dark\n(L1)\nNothing', fillcolor=colors['l1'])
        c.node('l1_scan', 'scan\n(L1)\nDust corrected', fillcolor=colors['l1'])
        c.node('l1_flat', 'flat\n(L1)\nDust corrected', fillcolor=colors['l1'])
        c.node('l1_flat_center', 'flat_center\n(L1)\nDust corrected', fillcolor=colors['l1'])
    
    # L2 data
    with dot.subgraph(name='cluster_l2') as c:
        c.attr(label='L2 (_l2.fits)', style='rounded', bgcolor=colors['l2'])
        c.node('l2_dark', 'dark\n(L2)\nNothing', fillcolor=colors['l2'])
        c.node('l2_scan', 'scan\n(L2)\nY-shift corrected', fillcolor=colors['l2'])
        c.node('l2_flat', 'flat\n(L2)\nY-shift corrected', fillcolor=colors['l2'])
        c.node('l2_flat_center', 'flat_center\n(L2)\nY-shift corrected', fillcolor=colors['l2'])
    
    # L3 data
    with dot.subgraph(name='cluster_l3') as c:
        c.attr(label='L3 (_l3.fits)', style='rounded', bgcolor=colors['l3'])
        c.node('l3_dark', 'dark\n(L3)\nNothing', fillcolor=colors['l3'])
        c.node('l3_scan', 'scan\n(L3)\nDesmiled', fillcolor=colors['l3'])
        c.node('l3_flat', 'flat\n(L3)\nDesmiled', fillcolor=colors['l3'])
        c.node('l3_flat_center', 'flat_center\n(L3)\nDesmiled', fillcolor=colors['l3'])
    
    # =========================================================================
    # AUXILIARY FILES
    # =========================================================================
    
    with dot.subgraph(name='cluster_aux') as c:
        c.attr(label='Auxiliary Files', style='dashed,rounded', bgcolor=colors['auxiliary'])
        c.node('aux_dust_flat', 'dust_flat\n(upper/lower)', shape='note', fillcolor=colors['auxiliary'])
        c.node('aux_yshift', 'y_shift arrays\n(upper/lower)', shape='note', fillcolor=colors['auxiliary'])
        c.node('aux_offset_map', 'offset_map\n(upper/lower)', shape='note', fillcolor=colors['auxiliary'])
        c.node('aux_illum', 'illumination_pattern\n(upper/lower)', shape='note', fillcolor=colors['auxiliary'])
    
    # =========================================================================
    # FUNCTION NODES
    # =========================================================================
    
    dot.node('func_raw_to_l0', 'reduce_raw_to_l0\n• Stack frames\n• Combine upper/lower', 
             shape='box', style='filled', fillcolor=colors['function'])
    
    dot.node('func_l0_to_l1', 'reduce_l0_to_l1\n• Dust flat correction\n• spectroflat (flat_center only)', 
             shape='box', style='filled', fillcolor=colors['function'])
    
    dot.node('func_l1_to_l2', 'reduce_l1_to_l2\n• Y-shift correction\n• _apply_yshift_correction_to_half', 
             shape='box', style='filled', fillcolor=colors['function'])
    
    dot.node('func_l2_to_l3', 'reduce_l2_to_l3\n• Desmiling\n• _apply_desmiling\n• spectroflat (flat_center only)', 
             shape='box', style='filled', fillcolor=colors['function'])
    
    # =========================================================================
    # EDGES: RAW -> L0
    # =========================================================================
    
    dot.edge('raw_dark', 'func_raw_to_l0')
    dot.edge('raw_scan', 'func_raw_to_l0')
    dot.edge('raw_flat', 'func_raw_to_l0')
    dot.edge('raw_flat_center', 'func_raw_to_l0')
    
    dot.edge('func_raw_to_l0', 'l0_dark')
    dot.edge('func_raw_to_l0', 'l0_scan')
    dot.edge('func_raw_to_l0', 'l0_flat')
    dot.edge('func_raw_to_l0', 'l0_flat_center')
    
    # =========================================================================
    # EDGES: L0 -> L1
    # =========================================================================
    
    dot.edge('l0_dark', 'func_l0_to_l1')
    dot.edge('l0_scan', 'func_l0_to_l1')
    dot.edge('l0_flat', 'func_l0_to_l1')
    dot.edge('l0_flat_center', 'func_l0_to_l1')
    
    dot.edge('func_l0_to_l1', 'l1_dark')
    dot.edge('func_l0_to_l1', 'l1_scan')
    dot.edge('func_l0_to_l1', 'l1_flat')
    dot.edge('func_l0_to_l1', 'l1_flat_center')
    
    # Auxiliary: dust flat from flat_center L1
    dot.edge('func_l0_to_l1', 'aux_dust_flat', style='dashed', color='orange', 
             label='generates')
    dot.edge('aux_dust_flat', 'func_l0_to_l1', style='dashed', color='orange',
             label='uses')
    
    # =========================================================================
    # EDGES: L1 -> L2
    # =========================================================================
    
    dot.edge('l1_dark', 'func_l1_to_l2')
    dot.edge('l1_scan', 'func_l1_to_l2')
    dot.edge('l1_flat', 'func_l1_to_l2')
    dot.edge('l1_flat_center', 'func_l1_to_l2')
    
    dot.edge('func_l1_to_l2', 'l2_dark')
    dot.edge('func_l1_to_l2', 'l2_scan')
    dot.edge('func_l1_to_l2', 'l2_flat')
    dot.edge('func_l1_to_l2', 'l2_flat_center')
    
    # Auxiliary: y-shift arrays
    dot.edge('aux_yshift', 'func_l1_to_l2', style='dashed', color='blue',
             label='uses')
    
    # =========================================================================
    # EDGES: L2 -> L3
    # =========================================================================
    
    dot.edge('l2_dark', 'func_l2_to_l3')
    dot.edge('l2_scan', 'func_l2_to_l3')
    dot.edge('l2_flat', 'func_l2_to_l3')
    dot.edge('l2_flat_center', 'func_l2_to_l3')
    
    dot.edge('func_l2_to_l3', 'l3_dark')
    dot.edge('func_l2_to_l3', 'l3_scan')
    dot.edge('func_l2_to_l3', 'l3_flat')
    dot.edge('func_l2_to_l3', 'l3_flat_center')
    
    # Auxiliary: offset map and illumination pattern
    dot.edge('func_l2_to_l3', 'aux_offset_map', style='dashed', color='purple',
             label='generates\n(flat_center)')
    dot.edge('func_l2_to_l3', 'aux_illum', style='dashed', color='purple',
             label='generates\n(flat_center)')
    dot.edge('aux_offset_map', 'func_l2_to_l3', style='dashed', color='purple',
             label='uses')
    
    # =========================================================================
    # RENDER
    # =========================================================================
    
    output_file = output_path / 'pipeline_diagram'
    dot.render(str(output_file), format='png', cleanup=True)
    dot.render(str(output_file), format='pdf', cleanup=True)
    
    print(f"✓ Pipeline diagram saved to:")
    print(f"  {output_file}.png")
    print(f"  {output_file}.pdf")
    
    return dot


def create_detailed_pipeline_diagram(output_dir=directory+'pipeline'):
    """Create a more detailed diagram focusing on data flow per data type."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    dot = Digraph('THEMIS_Pipeline_Detailed', comment='THEMIS Data Reduction Pipeline - Detailed')
    dot.attr(rankdir='LR', splines='polyline', nodesep='0.3', ranksep='1.0')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
    
    colors = {
        'dark': '#D3D3D3',
        'scan': '#87CEEB',
        'flat': '#98FB98',
        'flat_center': '#FFB6C1',
        'aux': '#F0E68C',
    }
    
    levels = ['raw', 'L0', 'L1', 'L2', 'L3']
    data_types = ['dark', 'scan', 'flat', 'flat_center']
    
    # Create nodes for each level and data type
    for level in levels:
        with dot.subgraph(name=f'cluster_{level}') as c:
            c.attr(label=level, style='rounded', bgcolor='#F5F5F5')
            for dt in data_types:
                node_id = f'{level}_{dt}'
                c.node(node_id, dt, fillcolor=colors[dt])
    
    # Processing descriptions
    processing = {
        'raw_to_L0': {
            'dark': 'Stack frames',
            'scan': 'Stack frames',
            'flat': 'Stack frames',
            'flat_center': 'Stack frames',
        },
        'L0_to_L1': {
            'dark': 'Nothing',
            'scan': 'Dust flat correction\n(from flat_center)',
            'flat': 'Dust flat correction\n(from flat_center)',
            'flat_center': 'spectroflat\n→ dust_flat (aux)',
        },
        'L1_to_L2': {
            'dark': 'Nothing',
            'scan': 'Y-shift correction\n(from aux)',
            'flat': 'Y-shift correction\n(from aux)',
            'flat_center': 'Y-shift correction\n(from aux)',
        },
        'L2_to_L3': {
            'dark': 'Nothing',
            'scan': 'Desmiling\n(offset from flat_center)',
            'flat': 'Desmiling\n(offset from flat_center)',
            'flat_center': 'spectroflat\n→ offset_map (aux)\nDesmiling',
        },
    }
    
    # Create edges with labels
    for i in range(len(levels) - 1):
        from_level = levels[i]
        to_level = levels[i + 1]
        key = f'{from_level}_to_{to_level}'
        
        for dt in data_types:
            from_node = f'{from_level}_{dt}'
            to_node = f'{to_level}_{dt}'
            label = processing[key][dt]
            dot.edge(from_node, to_node, label=label, fontsize='8')
    
    # Auxiliary file dependencies (cross-data-type)
    dot.edge('L1_flat_center', 'L1_scan', style='dashed', color='orange', 
             label='dust_flat', constraint='false')
    dot.edge('L1_flat_center', 'L1_flat', style='dashed', color='orange',
             label='dust_flat', constraint='false')
    
    dot.edge('L3_flat_center', 'L3_scan', style='dashed', color='purple',
             label='offset_map', constraint='false')
    dot.edge('L3_flat_center', 'L3_flat', style='dashed', color='purple',
             label='offset_map', constraint='false')
    
    output_file = output_path / 'pipeline_diagram_detailed'
    dot.render(str(output_file), format='png', cleanup=True)
    dot.render(str(output_file), format='pdf', cleanup=True)
    
    print(f"✓ Detailed pipeline diagram saved to:")
    print(f"  {output_file}.png")
    print(f"  {output_file}.pdf")
    
    return dot


if __name__ == '__main__':
    print("="*70)
    print("THEMIS Data Reduction Pipeline Documentation")
    print("="*70)
    print()
    
    # Create both diagrams
    create_pipeline_diagram()
    print()
    create_detailed_pipeline_diagram()
    
    print()
    print("="*70)
    print("Pipeline Overview:")
    print("="*70)
    print("""
    RAW → L0: Stack frames, combine upper/lower halves
    L0  → L1: Dust flat correction (spectroflat for flat_center)
    L1  → L2: Y-shift correction using calibration target analysis
    L2  → L3: Desmiling using offset maps from flat_center
    
    Data Types:
    - dark:        Calibration dark frames (mostly passthrough)
    - scan:        Science observation scans (CycleSet, multiple frames)
    - flat:        Flat field data (FramesSet, single frame)
    - flat_center: Centered flat for calibration (FramesSet, generates aux files)
    
    Auxiliary Files:
    - dust_flat:           Generated at L1 from flat_center
    - y_shift arrays:      Used at L2 (from calibration target analysis)
    - offset_map:          Generated at L3 from flat_center (spectroflat)
    - illumination_pattern: Generated at L3 from flat_center (spectroflat)
    
    Key Functions:
    - reduce_raw_to_l0()   : Stack and combine raw frames
    - reduce_l0_to_l1()    : Apply dust flat correction
    - reduce_l1_to_l2()    : Apply y-shift correction
    - reduce_l2_to_l3()    : Apply desmiling (smile correction)
    - _apply_desmiling()   : Row-by-row smile correction using offset maps
    """)
