// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>

// AUTOMATICALLY GENERATED by occamygen.py; edit the script instead.


`include "axi/typedef.svh"

/// Occamy Stage 1 Quadrant
module occamy_quadrant_s1
  import occamy_pkg::*;
(
    input  logic                                             clk_i,
    input  logic                                             rst_ni,
    input  logic                                             test_mode_i,
    input  tile_id_t                                         tile_id_i,
    input  logic                     [NrCoresS1Quadrant-1:0] meip_i,
    input  logic                     [NrCoresS1Quadrant-1:0] mtip_i,
    input  logic                     [NrCoresS1Quadrant-1:0] msip_i,
    // Next-Level
    output axi_a48_d64_i4_u5_req_t                           quadrant_narrow_out_req_o,
    input  axi_a48_d64_i4_u5_resp_t                          quadrant_narrow_out_rsp_i,
    input  axi_a48_d64_i7_u5_req_t                           quadrant_narrow_in_req_i,
    output axi_a48_d64_i7_u5_resp_t                          quadrant_narrow_in_rsp_o,
    output axi_a48_d512_i4_u0_req_t                          quadrant_wide_out_req_o,
    input  axi_a48_d512_i4_u0_resp_t                         quadrant_wide_out_rsp_i,
    input  axi_a48_d512_i5_u0_req_t                          quadrant_wide_in_req_i,
    output axi_a48_d512_i5_u0_resp_t                         quadrant_wide_in_rsp_o,
    // SRAM configuration
    input  sram_cfg_quadrant_t                               sram_cfg_i
);

  // Calculate cluster base address based on `tile id`.
  addr_t [0:0] cluster_base_addr;
  assign cluster_base_addr[0] = ClusterBaseOffset + tile_id_i * NrClustersS1Quadrant * ClusterAddressSpace + 0 * ClusterAddressSpace;

  // Define types for IOTLBs
  `AXI_TLB_TYPEDEF_ALL(tlb, logic [AddrWidth-12-1:0], logic [AddrWidth-12-1:0])

  // Signals from Controller
  logic clk_quadrant, rst_quadrant_n;
  logic [3:0] isolate, isolated;
  logic ro_enable, ro_flush_valid, ro_flush_ready;
  logic [3:0][47:0] ro_start_addr, ro_end_addr;
  logic narrow_tlb_enable;
  tlb_entry_t [7:0] narrow_tlb_entries;
  logic wide_tlb_enable;
  tlb_entry_t [7:0] wide_tlb_entries;

  ///////////////////
  //   CROSSBARS   //
  ///////////////////

  /// Address map of the `wide_xbar_quadrant_s1` crossbar.
  xbar_rule_48_t [0:0] WideXbarQuadrantS1Addrmap;
  assign WideXbarQuadrantS1Addrmap = '{
  '{ idx: 1, start_addr: cluster_base_addr[0], end_addr: cluster_base_addr[0] + ClusterAddressSpace }
};

  wide_xbar_quadrant_s1_in_req_t   [1:0] wide_xbar_quadrant_s1_in_req;
  wide_xbar_quadrant_s1_in_resp_t  [1:0] wide_xbar_quadrant_s1_in_rsp;
  wide_xbar_quadrant_s1_out_req_t  [1:0] wide_xbar_quadrant_s1_out_req;
  wide_xbar_quadrant_s1_out_resp_t [1:0] wide_xbar_quadrant_s1_out_rsp;

  axi_xbar #(
      .Cfg          (WideXbarQuadrantS1Cfg),
      .Connectivity (4'b0110),
      .ATOPs        (0),
      .slv_aw_chan_t(axi_a48_d512_i3_u0_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d512_i4_u0_aw_chan_t),
      .w_chan_t     (axi_a48_d512_i3_u0_w_chan_t),
      .slv_b_chan_t (axi_a48_d512_i3_u0_b_chan_t),
      .mst_b_chan_t (axi_a48_d512_i4_u0_b_chan_t),
      .slv_ar_chan_t(axi_a48_d512_i3_u0_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d512_i4_u0_ar_chan_t),
      .slv_r_chan_t (axi_a48_d512_i3_u0_r_chan_t),
      .mst_r_chan_t (axi_a48_d512_i4_u0_r_chan_t),
      .slv_req_t    (axi_a48_d512_i3_u0_req_t),
      .slv_resp_t   (axi_a48_d512_i3_u0_resp_t),
      .mst_req_t    (axi_a48_d512_i4_u0_req_t),
      .mst_resp_t   (axi_a48_d512_i4_u0_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_wide_xbar_quadrant_s1 (
      .clk_i                (clk_quadrant),
      .rst_ni               (rst_quadrant_n),
      .test_i               (test_mode_i),
      .slv_ports_req_i      (wide_xbar_quadrant_s1_in_req),
      .slv_ports_resp_o     (wide_xbar_quadrant_s1_in_rsp),
      .mst_ports_req_o      (wide_xbar_quadrant_s1_out_req),
      .mst_ports_resp_i     (wide_xbar_quadrant_s1_out_rsp),
      .addr_map_i           (WideXbarQuadrantS1Addrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );

  /// Address map of the `narrow_xbar_quadrant_s1` crossbar.
  xbar_rule_48_t [0:0] NarrowXbarQuadrantS1Addrmap;
  assign NarrowXbarQuadrantS1Addrmap = '{
  '{ idx: 1, start_addr: cluster_base_addr[0], end_addr: cluster_base_addr[0] + ClusterAddressSpace }
};

  narrow_xbar_quadrant_s1_in_req_t   [1:0] narrow_xbar_quadrant_s1_in_req;
  narrow_xbar_quadrant_s1_in_resp_t  [1:0] narrow_xbar_quadrant_s1_in_rsp;
  narrow_xbar_quadrant_s1_out_req_t  [1:0] narrow_xbar_quadrant_s1_out_req;
  narrow_xbar_quadrant_s1_out_resp_t [1:0] narrow_xbar_quadrant_s1_out_rsp;

  axi_xbar #(
      .Cfg          (NarrowXbarQuadrantS1Cfg),
      .Connectivity (4'b0110),
      .ATOPs        (1),
      .slv_aw_chan_t(axi_a48_d64_i4_u5_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d64_i5_u5_aw_chan_t),
      .w_chan_t     (axi_a48_d64_i4_u5_w_chan_t),
      .slv_b_chan_t (axi_a48_d64_i4_u5_b_chan_t),
      .mst_b_chan_t (axi_a48_d64_i5_u5_b_chan_t),
      .slv_ar_chan_t(axi_a48_d64_i4_u5_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d64_i5_u5_ar_chan_t),
      .slv_r_chan_t (axi_a48_d64_i4_u5_r_chan_t),
      .mst_r_chan_t (axi_a48_d64_i5_u5_r_chan_t),
      .slv_req_t    (axi_a48_d64_i4_u5_req_t),
      .slv_resp_t   (axi_a48_d64_i4_u5_resp_t),
      .mst_req_t    (axi_a48_d64_i5_u5_req_t),
      .mst_resp_t   (axi_a48_d64_i5_u5_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_narrow_xbar_quadrant_s1 (
      .clk_i                (clk_quadrant),
      .rst_ni               (rst_quadrant_n),
      .test_i               (test_mode_i),
      .slv_ports_req_i      (narrow_xbar_quadrant_s1_in_req),
      .slv_ports_resp_o     (narrow_xbar_quadrant_s1_in_rsp),
      .mst_ports_req_o      (narrow_xbar_quadrant_s1_out_req),
      .mst_ports_resp_i     (narrow_xbar_quadrant_s1_out_rsp),
      .addr_map_i           (NarrowXbarQuadrantS1Addrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );


  ///////////////////////////////
  // Narrow In + IW Converter //
  ///////////////////////////////
  axi_a48_d64_i7_u5_req_t  narrow_cluster_in_ctrl_req;
  axi_a48_d64_i7_u5_resp_t narrow_cluster_in_ctrl_rsp;

  axi_a48_d64_i7_u5_req_t  narrow_cluster_in_ctrl_cut_req;
  axi_a48_d64_i7_u5_resp_t narrow_cluster_in_ctrl_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d64_i7_u5_aw_chan_t),
      .w_chan_t(axi_a48_d64_i7_u5_w_chan_t),
      .b_chan_t(axi_a48_d64_i7_u5_b_chan_t),
      .ar_chan_t(axi_a48_d64_i7_u5_ar_chan_t),
      .r_chan_t(axi_a48_d64_i7_u5_r_chan_t),
      .axi_req_t(axi_a48_d64_i7_u5_req_t),
      .axi_resp_t(axi_a48_d64_i7_u5_resp_t)
  ) i_narrow_cluster_in_ctrl_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_in_ctrl_req),
      .slv_resp_o(narrow_cluster_in_ctrl_rsp),
      .mst_req_o(narrow_cluster_in_ctrl_cut_req),
      .mst_resp_i(narrow_cluster_in_ctrl_cut_rsp)
  );
  axi_a48_d64_i7_u5_req_t  narrow_cluster_in_isolate_req;
  axi_a48_d64_i7_u5_resp_t narrow_cluster_in_isolate_rsp;

  axi_isolate #(
      .NumPending(32),
      .TerminateTransaction(1),
      .AtopSupport(1),
      .AxiIdWidth(7),
      .AxiAddrWidth(48),
      .AxiDataWidth(64),
      .AxiUserWidth(5),
      .axi_req_t(axi_a48_d64_i7_u5_req_t),
      .axi_resp_t(axi_a48_d64_i7_u5_resp_t)
  ) i_narrow_cluster_in_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_in_ctrl_cut_req),
      .slv_resp_o(narrow_cluster_in_ctrl_cut_rsp),
      .mst_req_o(narrow_cluster_in_isolate_req),
      .mst_resp_i(narrow_cluster_in_isolate_rsp),
      .isolate_i(isolate[0]),
      .isolated_o(isolated[0])
  );

  axi_id_remap #(
      .AxiSlvPortIdWidth(7),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d64_i7_u5_req_t),
      .slv_resp_t(axi_a48_d64_i7_u5_resp_t),
      .mst_req_t(axi_a48_d64_i4_u5_req_t),
      .mst_resp_t(axi_a48_d64_i4_u5_resp_t)
  ) i_narrow_cluster_in_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_cluster_in_isolate_req),
      .slv_resp_o(narrow_cluster_in_isolate_rsp),
      .mst_req_o(narrow_xbar_quadrant_s1_in_req[NARROW_XBAR_QUADRANT_S1_IN_TOP]),
      .mst_resp_i(narrow_xbar_quadrant_s1_in_rsp[NARROW_XBAR_QUADRANT_S1_IN_TOP])
  );


  /////////////////////////////////////
  // Narrow Out + TLB + IW Converter //
  /////////////////////////////////////
  axi_a48_d64_i5_u5_req_t  narrow_cluster_out_tlb_req;
  axi_a48_d64_i5_u5_resp_t narrow_cluster_out_tlb_rsp;

  axi_tlb #(
      .AxiSlvPortAddrWidth(48),
      .AxiMstPortAddrWidth(48),
      .AxiDataWidth(64),
      .AxiIdWidth(5),
      .AxiUserWidth(5),
      .AxiSlvPortMaxTxns(32),
      .L1NumEntries(8),
      .L1CutAx(1'b1),
      .slv_req_t(axi_a48_d64_i5_u5_req_t),
      .mst_req_t(axi_a48_d64_i5_u5_req_t),
      .axi_resp_t(axi_a48_d64_i5_u5_resp_t),
      .entry_t(tlb_entry_t)
  ) i_narrow_cluster_out_tlb (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .test_en_i(test_mode_i),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_TOP]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_TOP]),
      .mst_req_o(narrow_cluster_out_tlb_req),
      .mst_resp_i(narrow_cluster_out_tlb_rsp),
      .entries_i(narrow_tlb_entries),
      .bypass_i(~narrow_tlb_enable)
  );

  axi_a48_d64_i4_u5_req_t  narrow_cluster_out_iwc_req;
  axi_a48_d64_i4_u5_resp_t narrow_cluster_out_iwc_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d64_i5_u5_req_t),
      .slv_resp_t(axi_a48_d64_i5_u5_resp_t),
      .mst_req_t(axi_a48_d64_i4_u5_req_t),
      .mst_resp_t(axi_a48_d64_i4_u5_resp_t)
  ) i_narrow_cluster_out_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_cluster_out_tlb_req),
      .slv_resp_o(narrow_cluster_out_tlb_rsp),
      .mst_req_o(narrow_cluster_out_iwc_req),
      .mst_resp_i(narrow_cluster_out_iwc_rsp)
  );
  axi_a48_d64_i4_u5_req_t  narrow_cluster_out_isolate_req;
  axi_a48_d64_i4_u5_resp_t narrow_cluster_out_isolate_rsp;

  axi_isolate #(
      .NumPending(32),
      .TerminateTransaction(0),
      .AtopSupport(1),
      .AxiIdWidth(4),
      .AxiAddrWidth(48),
      .AxiDataWidth(64),
      .AxiUserWidth(5),
      .axi_req_t(axi_a48_d64_i4_u5_req_t),
      .axi_resp_t(axi_a48_d64_i4_u5_resp_t)
  ) i_narrow_cluster_out_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_out_iwc_req),
      .slv_resp_o(narrow_cluster_out_iwc_rsp),
      .mst_req_o(narrow_cluster_out_isolate_req),
      .mst_resp_i(narrow_cluster_out_isolate_rsp),
      .isolate_i(isolate[1]),
      .isolated_o(isolated[1])
  );

  axi_a48_d64_i4_u5_req_t  narrow_cluster_out_ctrl_req;
  axi_a48_d64_i4_u5_resp_t narrow_cluster_out_ctrl_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d64_i4_u5_aw_chan_t),
      .w_chan_t(axi_a48_d64_i4_u5_w_chan_t),
      .b_chan_t(axi_a48_d64_i4_u5_b_chan_t),
      .ar_chan_t(axi_a48_d64_i4_u5_ar_chan_t),
      .r_chan_t(axi_a48_d64_i4_u5_r_chan_t),
      .axi_req_t(axi_a48_d64_i4_u5_req_t),
      .axi_resp_t(axi_a48_d64_i4_u5_resp_t)
  ) i_narrow_cluster_out_ctrl (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(narrow_cluster_out_isolate_req),
      .slv_resp_o(narrow_cluster_out_isolate_rsp),
      .mst_req_o(narrow_cluster_out_ctrl_req),
      .mst_resp_i(narrow_cluster_out_ctrl_rsp)
  );


  /////////////////////////////////////////
  // Wide Out + RO Cache + IW Converter  //
  /////////////////////////////////////////
  axi_a48_d512_i4_u0_req_t  wide_cluster_out_tlb_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_tlb_rsp;

  axi_tlb #(
      .AxiSlvPortAddrWidth(48),
      .AxiMstPortAddrWidth(48),
      .AxiDataWidth(512),
      .AxiIdWidth(4),
      .AxiUserWidth(1),
      .AxiSlvPortMaxTxns(32),
      .L1NumEntries(8),
      .L1CutAx(1'b1),
      .slv_req_t(axi_a48_d512_i4_u0_req_t),
      .mst_req_t(axi_a48_d512_i4_u0_req_t),
      .axi_resp_t(axi_a48_d512_i4_u0_resp_t),
      .entry_t(tlb_entry_t)
  ) i_wide_cluster_out_tlb (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .test_en_i(test_mode_i),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_TOP]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_TOP]),
      .mst_req_o(wide_cluster_out_tlb_req),
      .mst_resp_i(wide_cluster_out_tlb_rsp),
      .entries_i(wide_tlb_entries),
      .bypass_i(~wide_tlb_enable)
  );

  axi_a48_d512_i5_u0_req_t  snitch_ro_cache_req;
  axi_a48_d512_i5_u0_resp_t snitch_ro_cache_rsp;

  snitch_read_only_cache #(
      .LineWidth(1024),
      .LineCount(128),
      .SetCount(2),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiIdWidth(4),
      .AxiUserWidth(1),
      .MaxTrans(32),
      .NrAddrRules(4),
      .slv_req_t(axi_a48_d512_i4_u0_req_t),
      .slv_rsp_t(axi_a48_d512_i4_u0_resp_t),
      .mst_req_t(axi_a48_d512_i5_u0_req_t),
      .mst_rsp_t(axi_a48_d512_i5_u0_resp_t),
      .sram_cfg_data_t(sram_cfg_t),
      .sram_cfg_tag_t(sram_cfg_t)
  ) i_snitch_ro_cache (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .enable_i(ro_enable),
      .flush_valid_i(ro_flush_valid),
      .flush_ready_o(ro_flush_ready),
      .start_addr_i(ro_start_addr),
      .end_addr_i(ro_end_addr),
      .axi_slv_req_i(wide_cluster_out_tlb_req),
      .axi_slv_rsp_o(wide_cluster_out_tlb_rsp),
      .axi_mst_req_o(snitch_ro_cache_req),
      .axi_mst_rsp_i(snitch_ro_cache_rsp),
      .sram_cfg_data_i(sram_cfg_i.rocache_data),
      .sram_cfg_tag_i(sram_cfg_i.rocache_tag)
  );

  axi_a48_d512_i5_u0_req_t  snitch_ro_cache_cut_req;
  axi_a48_d512_i5_u0_resp_t snitch_ro_cache_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i5_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i5_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i5_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i5_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i5_u0_r_chan_t),
      .axi_req_t(axi_a48_d512_i5_u0_req_t),
      .axi_resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_snitch_ro_cache_cut (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(snitch_ro_cache_req),
      .slv_resp_o(snitch_ro_cache_rsp),
      .mst_req_o(snitch_ro_cache_cut_req),
      .mst_resp_i(snitch_ro_cache_cut_rsp)
  );
  axi_a48_d512_i4_u0_req_t  wide_cluster_out_iwc_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_iwc_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(32),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i4_u0_req_t),
      .mst_resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(snitch_ro_cache_cut_req),
      .slv_resp_o(snitch_ro_cache_cut_rsp),
      .mst_req_o(wide_cluster_out_iwc_req),
      .mst_resp_i(wide_cluster_out_iwc_rsp)
  );
  axi_a48_d512_i4_u0_req_t  wide_cluster_out_isolate_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_isolate_rsp;

  axi_isolate #(
      .NumPending(32),
      .TerminateTransaction(0),
      .AtopSupport(0),
      .AxiIdWidth(4),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiUserWidth(1),
      .axi_req_t(axi_a48_d512_i4_u0_req_t),
      .axi_resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_out_iwc_req),
      .slv_resp_o(wide_cluster_out_iwc_rsp),
      .mst_req_o(wide_cluster_out_isolate_req),
      .mst_resp_i(wide_cluster_out_isolate_rsp),
      .isolate_i(isolate[3]),
      .isolated_o(isolated[3])
  );

  axi_a48_d512_i4_u0_req_t  wide_cluster_out_isolate_cut_req;
  axi_a48_d512_i4_u0_resp_t wide_cluster_out_isolate_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i4_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i4_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i4_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i4_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i4_u0_r_chan_t),
      .axi_req_t(axi_a48_d512_i4_u0_req_t),
      .axi_resp_t(axi_a48_d512_i4_u0_resp_t)
  ) i_wide_cluster_out_isolate_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_out_isolate_req),
      .slv_resp_o(wide_cluster_out_isolate_rsp),
      .mst_req_o(wide_cluster_out_isolate_cut_req),
      .mst_resp_i(wide_cluster_out_isolate_cut_rsp)
  );


  assign quadrant_wide_out_req_o = wide_cluster_out_isolate_cut_req;
  assign wide_cluster_out_isolate_cut_rsp = quadrant_wide_out_rsp_i;

  ////////////////////////////
  // Wide In + IW Converter //
  ////////////////////////////
  axi_a48_d512_i5_u0_req_t  wide_cluster_in_iwc_req;
  axi_a48_d512_i5_u0_resp_t wide_cluster_in_iwc_rsp;

  axi_a48_d512_i5_u0_req_t  wide_cluster_in_iwc_cut_req;
  axi_a48_d512_i5_u0_resp_t wide_cluster_in_iwc_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i5_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i5_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i5_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i5_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i5_u0_r_chan_t),
      .axi_req_t(axi_a48_d512_i5_u0_req_t),
      .axi_resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_wide_cluster_in_iwc_cut (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_in_iwc_req),
      .slv_resp_o(wide_cluster_in_iwc_rsp),
      .mst_req_o(wide_cluster_in_iwc_cut_req),
      .mst_resp_i(wide_cluster_in_iwc_cut_rsp)
  );
  axi_a48_d512_i5_u0_req_t  wide_cluster_in_isolate_req;
  axi_a48_d512_i5_u0_resp_t wide_cluster_in_isolate_rsp;

  axi_isolate #(
      .NumPending(32),
      .TerminateTransaction(1),
      .AtopSupport(0),
      .AxiIdWidth(5),
      .AxiAddrWidth(48),
      .AxiDataWidth(512),
      .AxiUserWidth(1),
      .axi_req_t(axi_a48_d512_i5_u0_req_t),
      .axi_resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_wide_cluster_in_isolate (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .slv_req_i(wide_cluster_in_iwc_cut_req),
      .slv_resp_o(wide_cluster_in_iwc_cut_rsp),
      .mst_req_o(wide_cluster_in_isolate_req),
      .mst_resp_i(wide_cluster_in_isolate_rsp),
      .isolate_i(isolate[2]),
      .isolated_o(isolated[2])
  );

  axi_a48_d512_i5_u0_req_t  wide_cluster_in_isolate_cut_req;
  axi_a48_d512_i5_u0_resp_t wide_cluster_in_isolate_cut_rsp;

  axi_multicut #(
      .NoCuts(1),
      .aw_chan_t(axi_a48_d512_i5_u0_aw_chan_t),
      .w_chan_t(axi_a48_d512_i5_u0_w_chan_t),
      .b_chan_t(axi_a48_d512_i5_u0_b_chan_t),
      .ar_chan_t(axi_a48_d512_i5_u0_ar_chan_t),
      .r_chan_t(axi_a48_d512_i5_u0_r_chan_t),
      .axi_req_t(axi_a48_d512_i5_u0_req_t),
      .axi_resp_t(axi_a48_d512_i5_u0_resp_t)
  ) i_wide_cluster_in_isolate_cut (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_cluster_in_isolate_req),
      .slv_resp_o(wide_cluster_in_isolate_rsp),
      .mst_req_o(wide_cluster_in_isolate_cut_req),
      .mst_resp_i(wide_cluster_in_isolate_cut_rsp)
  );
  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(8),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(3),
      .slv_req_t(axi_a48_d512_i5_u0_req_t),
      .slv_resp_t(axi_a48_d512_i5_u0_resp_t),
      .mst_req_t(axi_a48_d512_i3_u0_req_t),
      .mst_resp_t(axi_a48_d512_i3_u0_resp_t)
  ) i_wide_cluster_in_iwc (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_cluster_in_isolate_cut_req),
      .slv_resp_o(wide_cluster_in_isolate_cut_rsp),
      .mst_req_o(wide_xbar_quadrant_s1_in_req[WIDE_XBAR_QUADRANT_S1_IN_TOP]),
      .mst_resp_i(wide_xbar_quadrant_s1_in_rsp[WIDE_XBAR_QUADRANT_S1_IN_TOP])
  );

  assign wide_cluster_in_iwc_req = quadrant_wide_in_req_i;
  assign quadrant_wide_in_rsp_o  = wide_cluster_in_iwc_rsp;

  /////////////////////////
  // Quadrant Controller //
  /////////////////////////

  occamy_quadrant_s1_ctrl #(
      .tlb_entry_t(tlb_entry_t)
  ) i_occamy_quadrant_s1_ctrl (
      .clk_i,
      .rst_ni,
      .test_mode_i,
      .tile_id_i,
      .clk_quadrant_o(clk_quadrant),
      .rst_quadrant_no(rst_quadrant_n),
      .isolate_o(isolate),
      .isolated_i(isolated),
      .ro_enable_o(ro_enable),
      .ro_flush_valid_o(ro_flush_valid),
      .ro_flush_ready_i(ro_flush_ready),
      .ro_start_addr_o(ro_start_addr),
      .ro_end_addr_o(ro_end_addr),
      .soc_out_req_o(quadrant_narrow_out_req_o),
      .soc_out_rsp_i(quadrant_narrow_out_rsp_i),
      .soc_in_req_i(quadrant_narrow_in_req_i),
      .soc_in_rsp_o(quadrant_narrow_in_rsp_o),
      .narrow_tlb_entries_o(narrow_tlb_entries),
      .narrow_tlb_enable_o(narrow_tlb_enable),
      .wide_tlb_entries_o(wide_tlb_entries),
      .wide_tlb_enable_o(wide_tlb_enable),
      .quadrant_out_req_o(narrow_cluster_in_ctrl_req),
      .quadrant_out_rsp_i(narrow_cluster_in_ctrl_rsp),
      .quadrant_in_req_i(narrow_cluster_out_ctrl_req),
      .quadrant_in_rsp_o(narrow_cluster_out_ctrl_rsp)
  );

  ///////////////
  // Cluster 0 //
  ///////////////
  axi_a48_d64_i2_u5_req_t  narrow_in_iwc_0_req;
  axi_a48_d64_i2_u5_resp_t narrow_in_iwc_0_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(5),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(2),
      .slv_req_t(axi_a48_d64_i5_u5_req_t),
      .slv_resp_t(axi_a48_d64_i5_u5_resp_t),
      .mst_req_t(axi_a48_d64_i2_u5_req_t),
      .mst_resp_t(axi_a48_d64_i2_u5_resp_t)
  ) i_narrow_in_iwc_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(narrow_xbar_quadrant_s1_out_req[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .slv_resp_o(narrow_xbar_quadrant_s1_out_rsp[NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .mst_req_o(narrow_in_iwc_0_req),
      .mst_resp_i(narrow_in_iwc_0_rsp)
  );
  axi_a48_d64_i4_u5_req_t  narrow_out_0_req;
  axi_a48_d64_i4_u5_resp_t narrow_out_0_rsp;

  assign narrow_xbar_quadrant_s1_in_req[NARROW_XBAR_QUADRANT_S1_IN_CLUSTER_0] = narrow_out_0_req;
  assign narrow_out_0_rsp = narrow_xbar_quadrant_s1_in_rsp[NARROW_XBAR_QUADRANT_S1_IN_CLUSTER_0];

  axi_a48_d512_i1_u0_req_t  wide_in_iwc_0_req;
  axi_a48_d512_i1_u0_resp_t wide_in_iwc_0_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(4),
      .AxiSlvPortMaxUniqIds(2),
      .AxiMaxTxnsPerId(32),
      .AxiMstPortIdWidth(1),
      .slv_req_t(axi_a48_d512_i4_u0_req_t),
      .slv_resp_t(axi_a48_d512_i4_u0_resp_t),
      .mst_req_t(axi_a48_d512_i1_u0_req_t),
      .mst_resp_t(axi_a48_d512_i1_u0_resp_t)
  ) i_wide_in_iwc_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .slv_req_i(wide_xbar_quadrant_s1_out_req[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .slv_resp_o(wide_xbar_quadrant_s1_out_rsp[WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_0]),
      .mst_req_o(wide_in_iwc_0_req),
      .mst_resp_i(wide_in_iwc_0_rsp)
  );
  axi_a48_d512_i3_u0_req_t  wide_out_0_req;
  axi_a48_d512_i3_u0_resp_t wide_out_0_rsp;

  assign wide_xbar_quadrant_s1_in_req[WIDE_XBAR_QUADRANT_S1_IN_CLUSTER_0] = wide_out_0_req;
  assign wide_out_0_rsp = wide_xbar_quadrant_s1_in_rsp[WIDE_XBAR_QUADRANT_S1_IN_CLUSTER_0];



  logic [9:0] hart_base_id_0;
  assign hart_base_id_0 = HartIdOffset + tile_id_i * NrCoresS1Quadrant + 0 * NrCoresCluster;

  occamy_cluster_wrapper i_occamy_cluster_0 (
      .clk_i(clk_quadrant),
      .rst_ni(rst_quadrant_n),
      .meip_i(meip_i[0*NrCoresCluster+:NrCoresCluster]),
      .mtip_i(mtip_i[0*NrCoresCluster+:NrCoresCluster]),
      .msip_i(msip_i[0*NrCoresCluster+:NrCoresCluster]),
      .hart_base_id_i(hart_base_id_0),
      .cluster_base_addr_i(cluster_base_addr[0]),
      .narrow_in_req_i(narrow_in_iwc_0_req),
      .narrow_in_resp_o(narrow_in_iwc_0_rsp),
      .narrow_out_req_o(narrow_out_0_req),
      .narrow_out_resp_i(narrow_out_0_rsp),
      .wide_out_req_o(wide_out_0_req),
      .wide_out_resp_i(wide_out_0_rsp),
      .wide_in_req_i(wide_in_iwc_0_req),
      .wide_in_resp_o(wide_in_iwc_0_rsp),
      .sram_cfgs_i(sram_cfg_i.cluster)
  );

endmodule
