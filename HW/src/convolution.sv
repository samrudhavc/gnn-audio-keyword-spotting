`timescale 1ns / 1ps

import graph_pkg::*;

module convolution #(
    parameter int PRECISION_IN               = graph_pkg::PRECISION_CONV1,
    parameter int PRECISION_OUT              = graph_pkg::PRECISION_CONV1,
    parameter int INPUT_DIM                  = 2,
    parameter int OUTPUT_DIM                 = 64,
    parameter int MULTIPLIER_DIFF_T          = 214742, //good
    parameter int ZERO_POINT_IN              = 0,  //good
    parameter int ZERO_POINT_OUT             = 36533, //good
    parameter int MULTIPLIER_OUT             = 58670, //good
    parameter int ZERO_POINT_WEIGHT          = 30075,
    parameter string INIT_PATH               = "???",
    parameter logic [PRECISION_IN-1:0] SCALE_IN [21:0]   = { 32767, 29490, 26214, 22937, 19660, 16383, 13107, 9830, 6553, 3277, 0, 65534,
                                                             62257, 58981, 55704, 52427, 49150, 45874, 42597, 39320, 36044, 32767 }
)(
    input logic clk,
    input logic reset,
    input event_type                   in_event,
    input edge_type  [MAX_EDGES-1:0]   in_edges,
    input logic [PRECISION_IN-1 :0]    in_features [INPUT_DIM-1 : 0],

    output event_type                  out_event,
    output edge_type  [MAX_EDGES-1:0]  out_edges,
    output logic [PRECISION_OUT-1 :0]  out_features [OUTPUT_DIM-1 : 0]
);

    logic [$clog2(F_RADIUS):0] counter, counter_reg, counter_read;
    logic [$clog2(F_RADIUS):0] counter_quant, counter_mul1, counter_mul2, counter_mul_out, counter_compare, counter_acc;
    logic [$clog2(OUTPUT_DIM/2):0] outdim_counter, outdim_counter_reg, outdim_counter_read, outdim_counter_quant;
    logic [$clog2(OUTPUT_DIM/2):0] outdim_counter_mul1, outdim_counter_mul2, outdim_counter_mul_out, outdim_counter_compare, outdim_counter_acc;
    event_type in_event_reg; // fifo output
    edge_type[MAX_EDGES-1:0] in_edges_reg;
    event_type out_event_reg; // fifo output
    edge_type[MAX_EDGES-1:0] out_edges_reg;
    typedef logic [PRECISION_IN-1 :0] features_type [INPUT_DIM-1 : 0];
    typedef logic [(PRECISION_IN*INPUT_DIM)-1 :0] memory_type;
    features_type in_features_reg;

    localparam IDLE = 2'd0;
    localparam CONV = 2'd1;
    
    localparam AWIDTH = $clog2(NUM_CHANNEL);
    localparam DWIDTH = INPUT_DIM * PRECISION_IN;
    logic state = IDLE;
    logic state_reg = IDLE;

    /////////////////////////////////////////////////////////////////
    //                        Handle MEMORY                        //
    /////////////////////////////////////////////////////////////////

    // Memory interface signals
    logic [AWIDTH-1:0] addra, addrb;
    logic [DWIDTH-1:0] dinb, douta, doutb;
    logic ena, wea, web, enb, ena_reg, enb_reg, web_reg;
    logic ena_quant, enb_quant, ena_mul1, enb_mul1, ena_mul2, enb_mul2, ena_mul_out, enb_mul_out;

    // Context memory instantiation
    memory #(
        .AWIDTH   ( AWIDTH  ),
        .DWIDTH   ( DWIDTH  ),
        .RAM_TYPE ( "block" )
    ) gen_memory (
        .clk      ( clk   ),
        .mem_ena  ( ena   ),    // READ only on PORTA
        .wea      ( wea   ),
        .addra    ( addra ),
        .dina     ( '0    ),
        .dinb     ( dinb  ),
        .douta    ( douta ),
        .mem_enb  ( enb   ),    // Write on last 
        .web      ( web   ),
        .addrb    ( addrb ),
        .doutb    ( doutb )
    );

    logic condition_a, condition_b, condition_a_reg, condition_b_reg;
    edge_type [MAX_EDGES-1:0] edges_reg;

    assign ena  = (counter_reg <= F_RADIUS && condition_a && state_reg==CONV) ? 1'b1 : 1'b0;
    assign enb  = (counter_reg <= F_RADIUS && condition_b && state_reg==CONV) ? 1'b1 : web;
    assign wea  = 0;
    assign dinb = memory_type'(in_features_reg);
    assign web  = (counter_reg == F_RADIUS) && state_reg==CONV;// && outdim_counter_reg==0;

    assign addra = in_event_reg.f + counter_reg*SKIP_STEP;
    assign addrb = in_event_reg.f - 100 + (counter_reg*SKIP_STEP);

    assign condition_a = in_edges[counter_reg].is_connected;
    assign condition_b = (counter_reg < F_RADIUS) ? in_edges[F_RADIUS+1+counter_reg].is_connected : 1'b0;
    logic start;

    // Counter and edge processing
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            start <= '0;
            ena_reg <= '0;
            enb_reg <= '0;
            web_reg <= '0;
            counter <= F_RADIUS;
            outdim_counter <= OUTPUT_DIM/2;
        end else begin
            if (state == CONV) begin
                outdim_counter <= outdim_counter + 1;
            end
            if (counter < F_RADIUS && state==CONV && outdim_counter == (OUTPUT_DIM/2)-1) begin
                counter <= counter + 1;
                outdim_counter <= '0;
            end
            if (in_event.valid) begin
                counter <= '0;
                outdim_counter <= '0;
                state <= CONV;
                in_event_reg <= in_event;
                in_edges_reg <= in_edges;
                in_features_reg <= in_features;
            end
            if (counter_reg == F_RADIUS && counter == F_RADIUS && state == CONV && outdim_counter == (OUTPUT_DIM/2)-1) begin
                state <= IDLE;
                start <= '1;
            end
            if (state == IDLE) begin
                if (start == 1) begin
                    out_edges_reg <= in_edges_reg;
                    out_event_reg <= in_event_reg;
                    start <= '0;
                end
                else begin
                    out_event_reg.valid <= '0;
                end
            end
            //if (counter_reg != counter_read) begin
                enb_reg <= enb;
                ena_reg <= ena;
                web_reg <= web;
            //end
            ena_quant <= ena_reg;
            ena_mul1 <= ena_quant;
            ena_mul2 <= ena_mul1;
            ena_mul_out <= ena_mul2;
            
            enb_quant <= enb_reg;
            enb_mul1 <= enb_quant;
            enb_mul2 <= enb_mul1;
            enb_mul_out <= enb_mul2;

            state_reg <= state;

            outdim_counter_reg <= outdim_counter;
            outdim_counter_read <= outdim_counter_reg;
            outdim_counter_quant <= outdim_counter_read;
            outdim_counter_mul1 <= outdim_counter_quant;
            outdim_counter_mul2 <= outdim_counter_mul1;
            outdim_counter_mul_out <= outdim_counter_mul2;
            outdim_counter_compare <= outdim_counter_mul_out;
            outdim_counter_acc <= outdim_counter_compare;

            counter_reg <= counter;
            counter_read <= counter_reg;
            counter_quant <= counter_read;
            counter_mul1 <= counter_quant;
            counter_mul2 <= counter_mul1;
            counter_mul_out <= counter_mul2;
            counter_compare <= counter_mul_out;
            counter_acc <= counter_compare;
        end
    end

    /////////////////////////////////////////////////////////////////
    //                      Quantize inputs                        //
    /////////////////////////////////////////////////////////////////

    logic signed [PRECISION_IN:0]   features_a_temp [INPUT_DIM-1:0];
    logic signed [PRECISION_IN:0]   features_b_temp [INPUT_DIM-1:0];
    logic signed [PRECISION_IN:0]   features_a [INPUT_DIM+1:0];
    logic signed [PRECISION_IN:0]   features_b [INPUT_DIM+1:0];

    genvar a, b;
    generate
        for (a = 0; a < INPUT_DIM; a++) begin : port_a_assign
            always @(posedge clk) begin
                features_a_temp[a][PRECISION_IN-1 : 0] = {douta[((PRECISION_IN)*(a+1))-1 : (PRECISION_IN*a)]};
                features_a_temp[a][PRECISION_IN] = 0;
                features_a[a+2] <= ena_reg ? features_a_temp[a]-ZERO_POINT_IN : '0;
            end
        end
        for (b = 0; b < INPUT_DIM; b++) begin : port_b_assign
            always @(posedge clk) begin
                features_b_temp[b][PRECISION_IN-1 : 0] = !web_reg ? {doutb[((PRECISION_IN)*(b+1))-1 : (PRECISION_IN*b)]}
                                                               : in_features_reg[b];
                features_b_temp[b][PRECISION_IN] = 0;
                features_b[b+2] <= enb_reg ? features_b_temp[b]-ZERO_POINT_IN : '0;
            end
        end       
    endgenerate

    always @(posedge clk) begin
        features_a[1] <= ena_reg ? ((in_edges_reg[counter_read].dt * MULTIPLIER_DIFF_T)>>>16) : '0; //dif_t
        features_a[0] <= ena_reg ? SCALE_IN[counter_read] : '0;
        features_b[1] <= (enb_reg && counter_read < F_RADIUS) ? ((in_edges_reg[F_RADIUS+1+counter_read].dt * MULTIPLIER_DIFF_T)>>>16) : '0; //dif_t
        features_b[0] <= (enb_reg) ? SCALE_IN[F_RADIUS+1+counter_read] : '0;
    end

    /////////////////////////////////////////////////////////////////
    //                   Perform multiplications                   //
    /////////////////////////////////////////////////////////////////

    //Prepare weights
    localparam WEIGHT_WIDTH = ((INPUT_DIM+2)*(PRECISION_OUT))+32; //bias
    logic [WEIGHT_WIDTH-1 : 0]     weight_mem1;
    logic [WEIGHT_WIDTH-1 : 0]     weight_mem2;
    logic signed [PRECISION_OUT:0] single_weight1_reg [INPUT_DIM+1:0];
    logic signed [31:0]            single_bias1_reg;
    logic signed [PRECISION_OUT:0] single_weight1 [INPUT_DIM+1:0];
    logic signed [31:0]            single_bias1;
    logic signed [PRECISION_OUT:0] single_weight2_reg [INPUT_DIM+1:0];
    logic signed [31:0]            single_bias2_reg;
    logic signed [PRECISION_OUT:0] single_weight2 [INPUT_DIM+1:0];
    logic signed [31:0]            single_bias2;

    dual_port_memory_weights #(
        .AWIDTH   ( $clog2(OUTPUT_DIM)               ),
        .DWIDTH   ( (PRECISION_OUT*(INPUT_DIM+2))+32 ),
        .STEP     ( 32                               ),
        .RAM_TYPE ( "block"                          ),
        .INIT_PATH ( INIT_PATH                       )
    ) weights_memory   (
        .clk      ( clk      ),
        .en       ( state == CONV   ),
        .addr     ( outdim_counter  ),
        .dout1    ( weight_mem1     ),
        .dout2    ( weight_mem2     )
    );

    genvar w;
    generate
        for (w = 0; w < INPUT_DIM+2; w++) begin : weights_assign
            always @(posedge clk) begin
                single_weight1_reg[w] <= weight_mem1[(((PRECISION_OUT)*(w+1))-1)+32 : ((PRECISION_OUT)*w)+32] - ZERO_POINT_WEIGHT;
                single_weight2_reg[w] <= weight_mem2[(((PRECISION_OUT)*(w+1))-1)+32 : ((PRECISION_OUT)*w)+32] - ZERO_POINT_WEIGHT;
            end
        end
    endgenerate

    always @(posedge clk) begin
        single_bias1_reg <= weight_mem1[31:0];
        single_bias2_reg <= weight_mem2[31:0];
        single_weight1 <= single_weight1_reg;
        single_bias1 <= single_bias1_reg;
        single_weight2 <= single_weight2_reg;
        single_bias2 <= single_bias2_reg;
    end

    logic [PRECISION_OUT-1:0] output_mat_a1;
    logic [PRECISION_OUT-1:0] output_mat_a2;
    logic [PRECISION_OUT-1:0] output_mat_b1;
    logic [PRECISION_OUT-1:0] output_mat_b2;
    logic [PRECISION_OUT-1:0] output_mat_a_full [OUTPUT_DIM-1:0];
    logic [PRECISION_OUT-1:0] output_mat_b_full [OUTPUT_DIM-1:0];
    logic [PRECISION_OUT-1:0] output_mat_full [OUTPUT_DIM-1:0];
    logic [PRECISION_OUT-1:0] output_features [OUTPUT_DIM-1:0];


    //Handle multiplications and outputs
    vector_multiplication #(
        .INPUT_DIM         ( INPUT_DIM+2    ),
        .MULTIPLIER        ( MULTIPLIER_OUT ),
        .ZERO_POINT        ( ZERO_POINT_OUT ),
        .PRECISION_IN      ( PRECISION_IN   ),
        .PRECISION_OUT     ( PRECISION_OUT  )
    ) mul_a_1 (
        .clk             ( clk             ),
        .reset           ( reset           ),
        .feature_matrix  ( features_a      ),
        .weight_matrix   ( single_weight1  ),
        .bias            ( single_bias1    ),
        .output_matrix   ( output_mat_a1   )
    );

    vector_multiplication #(
        .INPUT_DIM         ( INPUT_DIM+2    ),
        .MULTIPLIER        ( MULTIPLIER_OUT ),
        .ZERO_POINT        ( ZERO_POINT_OUT ),
        .PRECISION_IN      ( PRECISION_IN   ),
        .PRECISION_OUT     ( PRECISION_OUT  )
    ) mul_a_2 (
        .clk             ( clk             ),
        .reset           ( reset           ),
        .feature_matrix  ( features_a      ),
        .weight_matrix   ( single_weight2  ),
        .bias            ( single_bias2    ),
        .output_matrix   ( output_mat_a2   )
    );

    vector_multiplication #(
        .INPUT_DIM         ( INPUT_DIM+2    ),
        .MULTIPLIER        ( MULTIPLIER_OUT ),
        .ZERO_POINT        ( ZERO_POINT_OUT ),
        .PRECISION_IN      ( PRECISION_IN   ),
        .PRECISION_OUT     ( PRECISION_OUT  )
    ) mul_b_1 (
        .clk             ( clk             ),
        .reset           ( reset           ),
        .feature_matrix  ( features_b      ),
        .weight_matrix   ( single_weight1  ),
        .bias            ( single_bias1    ),
        .output_matrix   ( output_mat_b1   )
    );

    vector_multiplication #(
        .INPUT_DIM         ( INPUT_DIM+2    ),
        .MULTIPLIER        ( MULTIPLIER_OUT ),
        .ZERO_POINT        ( ZERO_POINT_OUT ),
        .PRECISION_IN      ( PRECISION_IN   ),
        .PRECISION_OUT     ( PRECISION_OUT  )
    ) mul_b_2 (
        .clk             ( clk             ),
        .reset           ( reset           ),
        .feature_matrix  ( features_b      ),
        .weight_matrix   ( single_weight2  ),
        .bias            ( single_bias2    ),
        .output_matrix   ( output_mat_b2   )
    );

    always @(posedge clk) begin
        output_mat_a_full[outdim_counter_mul_out] <= ena_mul_out ? output_mat_a1 : '0;
        output_mat_a_full[outdim_counter_mul_out+32] <= ena_mul_out ? output_mat_a2 : '0;
        output_mat_b_full[outdim_counter_mul_out] <= enb_mul_out ? output_mat_b1 : '0;
        output_mat_b_full[outdim_counter_mul_out+32] <= enb_mul_out ? output_mat_b2 : '0;
    end

    always @(posedge clk) begin
        output_mat_full[outdim_counter_compare] <= output_mat_a_full[outdim_counter_compare] > output_mat_b_full[outdim_counter_compare] ? output_mat_a_full[outdim_counter_compare]
                                                                                                                                   : output_mat_b_full[outdim_counter_compare];
        output_mat_full[outdim_counter_compare+32] <= output_mat_a_full[outdim_counter_compare+32] > output_mat_b_full[outdim_counter_compare+32] ? output_mat_a_full[outdim_counter_compare+32]
                                                                                                                                   : output_mat_b_full[outdim_counter_compare+32];

        if (outdim_counter_acc == 0 && counter_acc == 0) begin
            output_features <= '{default:ZERO_POINT_OUT};;
            output_features[outdim_counter_acc] <= ZERO_POINT_OUT >= output_mat_full[outdim_counter_acc] ? ZERO_POINT_OUT : output_mat_full[outdim_counter_acc];
            output_features[outdim_counter_acc+32] <= ZERO_POINT_OUT >= output_mat_full[outdim_counter_acc+32] ? ZERO_POINT_OUT : output_mat_full[outdim_counter_acc+32];
        end
        else begin
            output_features[outdim_counter_acc] <= output_features[outdim_counter_acc] > output_mat_full[outdim_counter_acc] ? output_features[outdim_counter_acc] : output_mat_full[outdim_counter_acc];
            output_features[outdim_counter_acc+32] <= output_features[outdim_counter_acc+32] > output_mat_full[outdim_counter_acc+32] ? output_features[outdim_counter_acc+32] : output_mat_full[outdim_counter_acc+32];
        end
        out_features <= output_features;
    end

    delay_module #(
        .N        ( 32 ),
        .DELAY    ( 8  )
    ) delay_event (
        .clk   ( clk     ),
        .idata ( {out_event_reg} ),
        .odata ( {out_event}     )
    );

    delay_module #(
        .N        ( 357 ),
        .DELAY    ( 8   )
    ) delay_edge (
        .clk   ( clk     ),
        .idata ( {out_edges_reg} ),
        .odata ( {out_edges}     )
    );

    // synthesis translate_off
    always @(posedge clk) begin
        if (state_reg != IDLE && in_event.valid) begin
            $display("DECREASE THE FIFO THROUGHPUT");
            $stop;
        end
    end
    // synthesis translate_on

endmodule

