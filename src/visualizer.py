import textwrap

def get_agent_graph_html(active_node: str) -> str:
    """
    Returns an HTML string containing an SVG visualization of the 3 agents,
    using modern circular progress bars for nodes and connected lines.
    """
    
    # Define primary theme color (Indigo / MagicUI style)
    color_active = "rgb(79, 70, 229)"
    color_completed = "rgb(99, 102, 241)" 
    color_pending = "rgba(255, 255, 255, 0.1)"
    
    # State flags
    r_status = "pending"
    g_status = "pending"
    s_status = "pending"
    c_status = "pending"
    
    line1_class = ""
    line1_color = color_pending
    line2_class = ""
    line2_color = color_pending
    line3_class = ""
    line3_color = color_pending
    
    if active_node == "research":
        r_status = "active"
    elif active_node == "grade":
        r_status = "completed"
        g_status = "active"
        line1_class = "marching"
        line1_color = color_active
    elif active_node == "synthesize":
        r_status = "completed"
        g_status = "completed"
        s_status = "active"
        line1_class = "solid"
        line1_color = color_completed
        line2_class = "marching"
        line2_color = color_active
    elif active_node == "critique":
        r_status = "completed"
        g_status = "completed"
        s_status = "completed"
        c_status = "active"
        line1_class = "solid"
        line1_color = color_completed
        line2_class = "solid"
        line2_color = color_completed
        line3_class = "marching"
        line3_color = color_active
    elif active_node == "rewrite":
        r_status = "completed"
        g_status = "completed"
        s_status = "active" # Back to synthesis
        c_status = "pending"
        line1_class = "solid"
        line1_color = color_completed
        line2_class = "solid"
        line2_color = color_completed
        line3_class = "marching-reverse"
        line3_color = "rgb(234, 179, 8)"
    elif active_node == "complete":
        r_status = "completed"
        g_status = "completed"
        s_status = "completed"
        c_status = "completed"
        line1_class = "solid"
        line1_color = color_completed
        line2_class = "solid"
        line2_color = color_completed
        line3_class = "solid"
        line3_color = color_completed

    def make_node(label, status):
        c = 251.32
        if status == "completed":
            offset = 0
            anim_class = ""
            glow = f"drop-shadow(0 0 8px {color_completed})"
            text_color = "white"
            stroke = color_completed
            target_val = 100
            start_val = 100
        elif status == "active":
            offset = c * 0.25  # 75% filled, spinning
            anim_class = "spin-anim"
            glow = f"drop-shadow(0 0 12px {color_active})"
            text_color = "white"
            stroke = color_active
            target_val = 75
            start_val = 0
        else: # pending
            offset = c
            anim_class = ""
            glow = "none"
            text_color = "rgba(255,255,255,0.3)"
            stroke = color_pending
            target_val = 0
            start_val = 0
            
        return f"""
        <div class="node">
            <svg width="100" height="100" viewBox="0 0 100 100" style="transform: rotate(-90deg); filter: {glow};">
                <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="6" />
                <circle class="{anim_class}" cx="50" cy="50" r="40" fill="none" stroke="{stroke}" stroke-width="6" 
                        stroke-dasharray="{c}" stroke-dashoffset="{offset}" stroke-linecap="round" />
            </svg>
            <div class="node-text" style="color: {text_color};">
                {label}
                <div style="font-size: 9px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; opacity: 0.8; margin-top: 4px; font-weight: 500;">
                    <span class="counter" data-target="{target_val}">{start_val}</span> / 100
                </div>
            </div>
        </div>
        """

    html = textwrap.dedent(f"""
    <style>
        .graph-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 200px;
            background-color: transparent;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin-top: 20px;
            margin-bottom: 40px;
        }}
        .graph-container {{
            position: relative;
            width: 800px;
            height: 100px;
        }}
        .connectors {{
            position: absolute;
            top: 0;
            left: 0;
            z-index: 0;
        }}
        .line-path {{
            stroke-width: 4;
            stroke-linecap: round;
            transition: stroke 0.5s ease;
        }}
        .line-path.marching {{
            stroke-dasharray: 12, 12;
            animation: march 1s linear infinite;
        }}
        .line-path.marching-reverse {{
            stroke-dasharray: 12, 12;
            animation: march-reverse 1s linear infinite;
        }}
        .line-path.solid {{
            stroke-dasharray: none;
        }}
        @keyframes march {{
            to {{ stroke-dashoffset: -24; }}
        }}
        @keyframes march-reverse {{
            to {{ stroke-dashoffset: 24; }}
        }}
        
        .node {{
            width: 100px;
            height: 100px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .node-text {{
            position: absolute;
            top: 110px;
            font-weight: 600;
            letter-spacing: 1.5px;
            font-size: 11px;
            text-transform: uppercase;
            text-align: center;
            width: 120px;
            transition: color 0.5s ease;
        }}
        
        .spin-anim {{
            transform-origin: 50px 50px;
            animation: spin 1.5s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    
    <div class="graph-wrapper">
        <div class="graph-container">
            <svg class="connectors" width="800" height="100">
                <!-- Research to Grade -->
                <line x1="100" y1="50" x2="300" y2="50" stroke="{line1_color}" class="line-path {line1_class}" />
                <!-- Grade to Synthesize -->
                <line x1="300" y1="50" x2="500" y2="50" stroke="{line2_color}" class="line-path {line2_class}" />
                <!-- Synthesize to Critique -->
                <line x1="500" y1="50" x2="700" y2="50" stroke="{line3_color}" class="line-path {line3_class}" />
            </svg>
            
            <div style="position: absolute; left: 50px; top: 0px; z-index: 1;">
                {make_node("RESEARCH", r_status)}
            </div>
            
            <div style="position: absolute; left: 250px; top: 0px; z-index: 1;">
                {make_node("GRADE", g_status)}
            </div>
            
            <div style="position: absolute; left: 450px; top: 0px; z-index: 1;">
                {make_node("SYNTHESIZE", s_status)}
            </div>
            
            <div style="position: absolute; left: 650px; top: 0px; z-index: 1;">
                {make_node("CRITIQUE", c_status)}
            </div>
        </div>
    </div>
    
    <script>
        const counters = document.querySelectorAll('.counter');
        const speed = 25; // Controls animation speed
        counters.forEach(counter => {{
            const target = +counter.getAttribute('data-target');
            const start = +counter.innerText;
            if (target > 0 && start === 0) {{
                const updateCount = () => {{
                    const count = +counter.innerText;
                    const inc = target / speed;
                    if (count < target) {{
                        counter.innerText = Math.ceil(count + inc);
                        setTimeout(updateCount, 40);
                    }} else {{
                        counter.innerText = target;
                    }}
                }};
                updateCount();
            }}
        }});
    </script>
    """)
    return html
