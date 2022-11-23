// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>

// AUTOMATICALLY GENERATED by genoccamy.py; edit the script instead.

module occamy_cva6
  import occamy_pkg::*;
(
    input  logic                          clk_i,
    input  logic                          rst_ni,
    input  logic                    [1:0] irq_i,
    input  logic                          ipi_i,
    input  logic                          time_irq_i,
    input  logic                          debug_req_i,
    output axi_a48_d64_i4_u5_req_t        axi_req_o,
    input  axi_a48_d64_i4_u5_resp_t       axi_resp_i,
    input  sram_cfg_cva6_t                sram_cfg_i
);

  axi_a48_d64_i4_u5_req_t  cva6_axi_req;
  axi_a48_d64_i4_u5_resp_t cva6_axi_rsp;

  axi_a48_d64_i4_u5_req_t  cva6_axi_cut_req;
  axi_a48_d64_i4_u5_resp_t cva6_axi_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d64_i4_u5_aw_chan_t),
      .w_chan_t(axi_a48_d64_i4_u5_w_chan_t),
      .b_chan_t(axi_a48_d64_i4_u5_b_chan_t),
      .ar_chan_t(axi_a48_d64_i4_u5_ar_chan_t),
      .r_chan_t(axi_a48_d64_i4_u5_r_chan_t),
      .axi_req_t(axi_a48_d64_i4_u5_req_t),
      .axi_resp_t(axi_a48_d64_i4_u5_resp_t)
  ) i_cva6_axi_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(cva6_axi_req),
      .slv_resp_o(cva6_axi_rsp),
      .mst_req_o(cva6_axi_cut_req),
      .mst_resp_i(cva6_axi_cut_rsp)
  );


  assign axi_req_o = cva6_axi_cut_req;
  assign cva6_axi_cut_rsp = axi_resp_i;

  // TODO(zarubaf): Derive from system parameters
  localparam ariane_pkg::ariane_cfg_t CVA6OccamyConfig = '{
      RASDepth: 2,
      BTBEntries: 32,
      BHTEntries: 128,
      // DRAM -- SPM, SPM -- Boot ROM, Boot ROM -- Debug Module
      NrNonIdempotentRules:
      3,
      NonIdempotentAddrBase: {64'd1879572480, 64'd16908288, 64'h1000},
      NonIdempotentLength: {64'd267911168, 64'd1862139904, 64'd16773120},
      NrExecuteRegionRules: 5,
      // DRAM, Boot ROM, SPM, Debug Module
      ExecuteRegionAddrBase: {
        64'h10_0000_0000, 64'h8000_0000, 64'd16777216, 64'd1879048192, 64'h0
      },
      ExecuteRegionLength: {64'h2_0000_0000, 64'h8000_0000, 64'd131072, 64'd524288, 64'h1000},
      // cached region
      NrCachedRegionRules:
      2,
      CachedRegionAddrBase: {64'h8000_0000, 64'd1879048192},
      CachedRegionLength: {(64'hff_ffff_ffff - 64'h8000_0000), 64'd524288},
      //  cache config
      Axi64BitCompliant:
      1'b1,
      SwapEndianess: 1'b0,
      // debug
      DmBaseAddress:
      64'h0,
      NrPMPEntries: 8
  };

  logic [1:0] irq;
  logic       ipi;
  logic       time_irq;
  logic       debug_req;

  sync #(
      .STAGES(2)
  ) i_sync_debug (
      .clk_i,
      .rst_ni,
      .serial_i(debug_req_i),
      .serial_o(debug_req)
  );
  sync #(
      .STAGES(2)
  ) i_sync_ipi (
      .clk_i,
      .rst_ni,
      .serial_i(ipi_i),
      .serial_o(ipi)
  );
  sync #(
      .STAGES(2)
  ) i_sync_time_irq (
      .clk_i,
      .rst_ni,
      .serial_i(time_irq_i),
      .serial_o(time_irq)
  );
  sync #(
      .STAGES(2)
  ) i_sync_irq_0 (
      .clk_i,
      .rst_ni,
      .serial_i(irq_i[0]),
      .serial_o(irq[0])
  );
  sync #(
      .STAGES(2)
  ) i_sync_irq_1 (
      .clk_i,
      .rst_ni,
      .serial_i(irq_i[1]),
      .serial_o(irq[1])
  );

  localparam logic [63:0] BootAddr = 'd16777216;


  ariane #(
      .ArianeCfg(CVA6OccamyConfig),
      .AxiAddrWidth(48),
      .AxiDataWidth(64),
      .AxiIdWidth(4),
      .AxiUserWidth(5),
      .axi_ar_chan_t(axi_a48_d64_i4_u5_ar_chan_t),
      .axi_aw_chan_t(axi_a48_d64_i4_u5_aw_chan_t),
      .axi_w_chan_t(axi_a48_d64_i4_u5_w_chan_t),
      .axi_req_t(axi_a48_d64_i4_u5_req_t),
      .axi_rsp_t(axi_a48_d64_i4_u5_resp_t),
      .sram_cfg_t(sram_cfg_t)
  ) i_cva6 (
      .clk_i,
      .rst_ni,
      .boot_addr_i(BootAddr),
      .hart_id_i(64'h0),
      .irq_i(irq),
      .ipi_i(ipi),
      .time_irq_i(time_irq),
      .debug_req_i(debug_req),
      .axi_req_o(cva6_axi_req),
      .axi_resp_i(cva6_axi_rsp),
      .sram_cfg_idata_i(sram_cfg_i.icache_data),
      .sram_cfg_itag_i(sram_cfg_i.icache_tag),
      .sram_cfg_ddata_i(sram_cfg_i.dcache_data),
      .sram_cfg_dtag_i(sram_cfg_i.dcache_tag),
      .sram_cfg_dvalid_dirty_i(sram_cfg_i.dcache_valid_dirty)
  );

endmodule
