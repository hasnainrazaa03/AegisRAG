def get_agent_graph_html(active_node: str) -> str:
    """
    Returns an HTML string containing an SVG visualization of the 3 agents.
    The active_node determines which node is glowing and which paths are animated.
    """
    
    # CSS Classes mapping
    res_class = "active" if active_node == "research" else ("completed" if active_node in ["synthesize", "critique", "complete"] else "idle")
    syn_class = "active" if active_node == "synthesize" else ("completed" if active_node in ["critique", "complete"] else "idle")
    cri_class = "active" if active_node == "critique" else ("completed" if active_node == "complete" else "idle")
    
    # Path animations
    path_r_s = "animated" if active_node in ["synthesize", "critique", "complete"] else "hidden"
    path_s_c = "animated" if active_node in ["critique", "complete"] else "hidden"
    path_c_r = "animated" if active_node == "rewrite" else "hidden"
    
    # Flash effect
    flash_html = "<div class='flash-overlay'></div>" if active_node == "complete" else ""

    html = f"""
    <style>
        .graph-container {{
            position: relative;
            width: 100%;
            height: 400px;
            background: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .node {{
            position: absolute;
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: #1E1E2E;
            border: 2px solid #3B4252;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #D8DEE9;
            font-family: sans-serif;
            font-weight: bold;
            font-size: 14px;
            transition: all 0.5s ease;
            z-index: 2;
        }}
        .node .icon {{
            font-size: 24px;
            margin-bottom: 5px;
        }}
        
        /* Node States */
        .node.idle {{
            opacity: 0.5;
            box-shadow: none;
        }}
        .node.active {{
            border-color: #88C0D0;
            background: #2E3440;
            box-shadow: 0 0 25px 5px rgba(136, 192, 208, 0.6);
            transform: scale(1.1);
            color: #ECEFF4;
            opacity: 1;
        }}
        .node.completed {{
            border-color: #A3BE8C;
            box-shadow: 0 0 15px 2px rgba(163, 190, 140, 0.4);
            opacity: 0.9;
        }}
        
        /* Specific Node Positions (Triangle) */
        #node-research {{ top: 40px; left: calc(50% - 50px); }}
        #node-synthesize {{ bottom: 60px; right: calc(25% - 50px); }}
        #node-critique {{ bottom: 60px; left: calc(25% - 50px); }}
        
        /* SVG Connector Lines */
        svg.connectors {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}
        .path-line {{
            fill: none;
            stroke: #4C566A;
            stroke-width: 4;
            stroke-dasharray: 10, 10;
        }}
        .path-line.animated {{
            stroke: #88C0D0;
            stroke-dasharray: 15, 15;
            animation: dash 1s linear infinite;
        }}
        
        @keyframes dash {{
            to {{
                stroke-dashoffset: -30;
            }}
        }}
        
        /* Flash Overlay on Complete */
        .flash-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(163, 190, 140, 0.2);
            z-index: 10;
            animation: flash 1.5s ease-out forwards;
            pointer-events: none;
        }}
        @keyframes flash {{
            0% {{ background: rgba(163, 190, 140, 0.8); }}
            100% {{ background: rgba(163, 190, 140, 0); }}
        }}
    </style>

    <div class="graph-container">
        {flash_html}
        
        <svg class="connectors">
            <!-- Research -> Synthesize -->
            <path d="M 50% 100 Q 75% 150, 75% 300" class="path-line {path_r_s}" />
            <!-- Synthesize -> Critique -->
            <path d="M 75% 340 L 25% 340" class="path-line {path_s_c}" />
            <!-- Critique -> Research (Rewrite) -->
            <path d="M 25% 300 Q 25% 150, 50% 100" class="path-line {path_c_r}" />
        </svg>

        <div id="node-research" class="node {res_class}">
            <div class="icon">🔍</div>
            Research
        </div>
        <div id="node-synthesize" class="node {syn_class}">
            <div class="icon">✍️</div>
            Synthesize
        </div>
        <div id="node-critique" class="node {cri_class}">
            <div class="icon">🧐</div>
            Critique
        </div>
    </div>
    """
    return html
