`timescale 1ns / 1ps

import graph_pkg::*;

parameter int DATA_NUM = 12118;

module tb_generate_graph;

    logic clk;
    logic reset;
    logic [T_WIDTH-1:0] t;
    logic [F_WIDTH-1:0] f;
    logic is_valid;
    event_type out_event;
    edge_type [MAX_EDGES-1:0] out_edges;
    logic [T_WIDTH+F_WIDTH-1 : 0] mem [DATA_NUM-1 : 0];
    logic empty;
    logic [T_WIDTH-1:0] t_feature;
    logic [F_WIDTH-1:0] f_feature;
    
    integer mem_index;
    
    initial begin
        $readmemb("/home/360/360.3-Stages/360.3.91-HN280727/gcn_audio/gcn_audio.sim/hw_input.mem", mem);
    end

    generate_graph uut (
        .clk(clk),
        .reset(reset),
        .t(t),
        .f(f),
        .is_valid(is_valid),
        .out_event(out_event),
        .out_edges(out_edges),
        .t_feature(t_feature),
        .f_feature(f_feature)
    );

    always begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    initial begin
        reset = 1;
        is_valid = 0;
        mem_index = 0;
        #20;
        reset = 0;
        empty = 0;
        #40;

        @(posedge clk);
        is_valid = 0;

        while (mem_index < DATA_NUM) begin 
            @(posedge clk);
                {t, f} = mem[mem_index]; 
                mem_index = mem_index + 1;
                is_valid = 1;
        end

        @(posedge clk);
        is_valid = 0;

        #1600000;
        $finish;
    end

endmodule
