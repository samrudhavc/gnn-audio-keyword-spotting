`timescale 1ns / 1ps

import graph_pkg::*;

parameter int DATA_NUM = 12118;

module tb_edges_gen;

    logic clk;
    logic reset;
    logic [T_WIDTH-1:0] t;
    logic [F_WIDTH-1:0] f;
    logic is_valid;
    graph_event_type event_to_normalize;
    graph_edge_type [MAX_EDGES-1:0] edges_to_normalize;
    logic [T_WIDTH+F_WIDTH-1 : 0] mem [DATA_NUM-1 : 0];
    logic empty;
    logic [T_WIDTH-1:0] t_feature_to_normalize;
    logic [F_WIDTH-1:0] f_feature_to_normalize;
    logic [N_WIDTH-1:0] n;
    
    integer outfile;
    integer mem_index;
    
    initial begin
        outfile = $fopen("/home/360/360.3-Stages/360.3.91-HN280727/gcn_audio/gcn_audio.sim/output_results.json", "w");
        $readmemb("/home/360/360.3-Stages/360.3.91-HN280727/gcn_audio/gcn_audio.sim/hw_input.mem", mem);
        $fwrite(outfile, "[\n");
    end

    edges_gen #(
    ) u_edges_gen (
        .clk           ( clk                    ),
        .reset         ( reset                  ),
        .t             ( t                      ),
        .f             ( f                      ),
        .is_valid      ( is_valid               ),
        .out_event     ( event_to_normalize     ),
        .out_edges     ( edges_to_normalize     ),
        .t_feature     ( t_feature_to_normalize ),
        .f_feature     ( f_feature_to_normalize ),
        .n             ( n                      ),
        .empty         ( empty                  )
    );

    always begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    always @(posedge clk) begin
        if (event_to_normalize.valid == 1'b1) begin
            $fwrite(outfile, "  {\n");
            $fwrite(outfile, "    \"pos_item\": {\"t\": %0d, \"f\": %0d},\n",
                (event_to_normalize.t < 0) ? -event_to_normalize.t : event_to_normalize.t,
                (event_to_normalize.f < 0) ? -event_to_normalize.f : event_to_normalize.f);
            
            $fwrite(outfile, "    \"edges\": [\n");
        
            // Write each edge structure to the output file
            for (int j=0; j<MAX_EDGES; j=j+1) begin
                if(edges_to_normalize[j].is_connected) begin
                    $fwrite(outfile, "      {\"t\": %0d, \"f\": %0d, \"dt\": %0d, \"df\": %0d}",
                        edges_to_normalize[j].t, 
                        edges_to_normalize[j].f, 
                        (edges_to_normalize[j].dt < 0) ? -edges_to_normalize[j].dt : edges_to_normalize[j].dt,
                        (edges_to_normalize[j].df < 0) ? -edges_to_normalize[j].df : edges_to_normalize[j].df);

                    if (j < MAX_EDGES - 1) begin
                        for (int k=j+1; k<MAX_EDGES; k=k+1) begin
                            if(edges_to_normalize[k].is_connected) begin
                                $fwrite(outfile, ",\n");
                                break;
                            end
                        end
                    end else begin
                        $fwrite(outfile, "\n");
                    end
                end
            end
            
            $fwrite(outfile, "    ],\n");

            $fwrite(outfile, "    \"average_edge\": {\"avg_t\": %0d, \"avg_f\": %0d}\n",
                t_feature_to_normalize, f_feature_to_normalize);

            $fwrite(outfile, "  },\n");
        end
    end

    // Reset generation
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

        #1400000;
        $finish;
    end
    
    final begin
        $fseek(outfile, -3, 2);
        $fwrite(outfile, "  }\n]\n"); 
        $fclose(outfile);
    end

endmodule
