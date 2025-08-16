class TerrainCfg():
    mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale = 0.1 # [m]
    vertical_scale = 0.005 # [m]
    border_size = 25 # [m]
    curriculum = False
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    # rough terrain only:
    selected = False # select a unique terrain type and pass all arguments
    terrain_kwargs = None # Dict of arguments for selected terrain
    max_init_terrain_level = 5 # starting curriculum state
    terrain_length = 8.
    terrain_width = 8.
    num_rows= 10 # number of terrain rows (levels)
    num_cols = 20 # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    # trimesh only:
    slope_treshold = 0.4 #0.75 # slopes above this threshold will be corrected to vertical surfaces