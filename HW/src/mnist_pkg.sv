package graph_pkg;


    parameter F_RADIUS   = 10; //Search radius will be F_RADIUS*SKIPSTEPS                    
    parameter T_RADIUS   = 20000;                 
    parameter MAX_EDGES  = (F_RADIUS*2) + 1; //the same F neighbour possible!;         
    parameter PRECISION_GEN  = 16;                    
    parameter PRECISION_CONV1  = 16;                    
    parameter PRECISION_CONV2  = 8;                    
    parameter PRECISION_CONV3  = 8;                    
    parameter PRECISION_CONV4  = 8;                    

    parameter T_WIDTH  = 20; //Max of 1000000
    parameter F_WIDTH  = 10; //Max of 700
    
    parameter NUM_CHANNEL = 700;

    parameter INPUT_PARAMETER = 2; 
    
    parameter INPUT_DIM_1 = 2; 
    parameter OUTPUT_DIM_1 = 64;
    parameter OUTPUT_DIM_2 = 64;
    parameter OUTPUT_DIM_3 = 64;
    parameter OUTPUT_DIM_4 = 64;

    parameter ZERO_POINT = '0;
    parameter MULTIPLIER = '0;
   
    parameter SKIP_STEP = 10;
    
    typedef struct packed {
      logic [T_WIDTH -1: 0] t;
      logic [F_WIDTH -1: 0] f;
      logic                 is_last;
      logic                 valid;
    } event_type;

    typedef struct packed {
      logic [PRECISION_GEN-1 : 0] dt;
      logic                       is_connected;
    } edge_type;

    parameter DELTA_T_WIDTH = 15; //max value of 20000
    parameter GEN_MULTIPLIER_T = 4295;
    parameter GEN_MULTIPLIER_F = 6135480;
    parameter GEN_ZERO_POINT = '0;

endpackage