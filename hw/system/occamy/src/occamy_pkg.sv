// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
// Florian Zaruba <zarubaf@iis.ee.ethz.ch>
//
// AUTOMATICALLY GENERATED by occamygen.py; edit the script instead.
// verilog_lint: waive-start line-length

`include "axi/typedef.svh"
`include "register_interface/typedef.svh"
`include "apb/typedef.svh"

package occamy_pkg;
  localparam int unsigned MaxTransaction = 16;

  // Re-exports
  localparam int unsigned AddrWidth = occamy_cluster_pkg::AddrWidth;
  localparam int unsigned NarrowUserWidth = occamy_cluster_pkg::NarrowUserWidth;
  localparam int unsigned WideUserWidth = occamy_cluster_pkg::WideUserWidth;

  localparam int unsigned NrClustersS1Quadrant = 1;
  localparam int unsigned NrCoresCluster = occamy_cluster_pkg::NrCores;
  localparam int unsigned NrCoresS1Quadrant = NrClustersS1Quadrant * NrCoresCluster;

  // Memory cut configurations: one per memory parameterization
  typedef occamy_cluster_pkg::sram_cfg_t sram_cfg_t;

  typedef struct packed {
    sram_cfg_t rocache_tag;
    sram_cfg_t rocache_data;
    occamy_cluster_pkg::sram_cfgs_t cluster;
  } sram_cfg_quadrant_t;

  typedef struct packed {
    sram_cfg_t dcache_valid_dirty;
    sram_cfg_t dcache_tag;
    sram_cfg_t dcache_data;
    sram_cfg_t icache_tag;
    sram_cfg_t icache_data;
  } sram_cfg_cva6_t;

  typedef struct packed {
    sram_cfg_t spm_narrow;
    sram_cfg_t spm_wide;
    sram_cfg_cva6_t cva6;
    sram_cfg_quadrant_t quadrant;
  } sram_cfgs_t;

  localparam int unsigned SramCfgWidth = $bits(sram_cfg_t);
  localparam int unsigned SramCfgCount = $bits(sram_cfgs_t) / SramCfgWidth;

  typedef struct packed {
    logic [3:0] timer;
    logic [31:0] gpio;
    logic uart;
    logic spim_error;
    logic spim_spi_event;
    logic i2c_fmt_watermark;
    logic i2c_rx_watermark;
    logic i2c_fmt_overflow;
    logic i2c_rx_overflow;
    logic i2c_nak;
    logic i2c_scl_interference;
    logic i2c_sda_interference;
    logic i2c_stretch_timeout;
    logic i2c_sda_unstable;
    logic i2c_trans_complete;
    logic i2c_tx_empty;
    logic i2c_tx_nonempty;
    logic i2c_tx_overflow;
    logic i2c_acq_overflow;
    logic i2c_ack_stop;
    logic i2c_host_timeout;
    logic ecc_narrow_uncorrectable;
    logic ecc_narrow_correctable;
    logic ecc_wide_uncorrectable;
    logic ecc_wide_correctable;
    // 4 programmable, 8 HBM (1x per channel)
    logic [12:0] ext_irq;
    logic zero;
  } occamy_interrupt_t;

  localparam logic [15:0] PartNum = 2;
  localparam logic [31:0] IDCode = (dm::DbgVersion013 << 28) | (PartNum << 12) | 32'h1;

  typedef logic [5:0] tile_id_t;

  typedef logic [AddrWidth-1:0] addr_t;
  typedef logic [NarrowUserWidth-1:0] user_narrow_t;
  typedef logic [WideUserWidth-1:0] user_wide_t;

  typedef struct packed {
    logic [31:0] idx;
    logic [47:0] start_addr;
    logic [47:0] end_addr;
  } xbar_rule_48_t;


  typedef xbar_rule_48_t xbar_rule_t;

  /// We reserve hartid `0` for CVA6.
  localparam logic [9:0] HartIdOffset = 1;
  /// The base offset for each cluster.
  localparam addr_t ClusterBaseOffset = 'h1000_0000;
  /// The address space set aside for each slave.
  localparam addr_t ClusterAddressSpace = 'h4_0000;
  /// The address space of a single S1 quadrant.
  localparam addr_t S1QuadrantAddressSpace = ClusterAddressSpace * NrClustersS1Quadrant;
  /// The base offset of the quadrant configuration region.
  localparam addr_t S1QuadrantCfgBaseOffset = 'hb00_0000;
  /// The address space set aside for the configuration of each slave.
  localparam addr_t S1QuadrantCfgAddressSpace = 'h1_0000;



  // AXI-Lite bus with 48 bit address and 64 bit data.
  `AXI_LITE_TYPEDEF_ALL(axi_lite_a48_d64, logic [47:0], logic [63:0], logic [7:0])

  /// Inputs of the `soc_axi_lite_periph_xbar` crossbar.
  typedef enum int {
    SOC_AXI_LITE_PERIPH_XBAR_IN_SOC,
    SOC_AXI_LITE_PERIPH_XBAR_IN_DEBUG,
    SOC_AXI_LITE_PERIPH_XBAR_NUM_INPUTS
  } soc_axi_lite_periph_xbar_inputs_e;

  /// Outputs of the `soc_axi_lite_periph_xbar` crossbar.
  typedef enum int {
    SOC_AXI_LITE_PERIPH_XBAR_OUT_SOC,
    SOC_AXI_LITE_PERIPH_XBAR_OUT_DEBUG,
    SOC_AXI_LITE_PERIPH_XBAR_NUM_OUTPUTS
  } soc_axi_lite_periph_xbar_outputs_e;

  /// Configuration of the `soc_axi_lite_periph_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t SocAxiLitePeriphXbarCfg = '{
      NoSlvPorts: SOC_AXI_LITE_PERIPH_XBAR_NUM_INPUTS,
      NoMstPorts: SOC_AXI_LITE_PERIPH_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 4,
      MaxMstTrans: 4,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 0,
      AxiIdUsedSlvPorts: 0,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 2
  };

  // AXI plugs of the `soc_axi_lite_periph_xbar` crossbar.

  typedef axi_lite_a48_d64_req_t soc_axi_lite_periph_xbar_in_req_t;
  typedef axi_lite_a48_d64_req_t soc_axi_lite_periph_xbar_out_req_t;
  typedef axi_lite_a48_d64_rsp_t soc_axi_lite_periph_xbar_in_rsp_t;
  typedef axi_lite_a48_d64_rsp_t soc_axi_lite_periph_xbar_out_rsp_t;
  typedef axi_lite_a48_d64_aw_chan_t soc_axi_lite_periph_xbar_in_aw_chan_t;
  typedef axi_lite_a48_d64_aw_chan_t soc_axi_lite_periph_xbar_out_aw_chan_t;
  typedef axi_lite_a48_d64_w_chan_t soc_axi_lite_periph_xbar_in_w_chan_t;
  typedef axi_lite_a48_d64_w_chan_t soc_axi_lite_periph_xbar_out_w_chan_t;
  typedef axi_lite_a48_d64_b_chan_t soc_axi_lite_periph_xbar_in_b_chan_t;
  typedef axi_lite_a48_d64_b_chan_t soc_axi_lite_periph_xbar_out_b_chan_t;
  typedef axi_lite_a48_d64_ar_chan_t soc_axi_lite_periph_xbar_in_ar_chan_t;
  typedef axi_lite_a48_d64_ar_chan_t soc_axi_lite_periph_xbar_out_ar_chan_t;
  typedef axi_lite_a48_d64_r_chan_t soc_axi_lite_periph_xbar_in_r_chan_t;
  typedef axi_lite_a48_d64_r_chan_t soc_axi_lite_periph_xbar_out_r_chan_t;

  // Register bus with 48 bit address and 32 bit data.
  `REG_BUS_TYPEDEF_ALL(reg_a48_d32, logic [47:0], logic [31:0], logic [3:0])

  /// Inputs of the `hbm_cfg_xbar` crossbar.
  typedef enum int {
    HBM_CFG_XBAR_IN_CFG,
    HBM_CFG_XBAR_NUM_INPUTS
  } hbm_cfg_xbar_inputs_e;

  /// Outputs of the `hbm_cfg_xbar` crossbar.
  typedef enum int {
    HBM_CFG_XBAR_OUT_TOP,
    HBM_CFG_XBAR_OUT_PHY,
    HBM_CFG_XBAR_OUT_SEQ,
    HBM_CFG_XBAR_OUT_CTRL,
    HBM_CFG_XBAR_NUM_OUTPUTS
  } hbm_cfg_xbar_outputs_e;

  /// Address map of the `hbm_cfg_xbar` crossbar.
  localparam xbar_rule_48_t [3:0] HbmCfgXbarAddrmap = '{
      '{idx: 0, start_addr: 48'h08000000, end_addr: 48'h08400000},
      '{idx: 1, start_addr: 48'h09000000, end_addr: 48'h09100000},
      '{idx: 2, start_addr: 48'h0a000000, end_addr: 48'h0a010000},
      '{idx: 3, start_addr: 48'h0a800000, end_addr: 48'h0a810000}
  };

  // AXI-Lite bus with 48 bit address and 32 bit data.
  `AXI_LITE_TYPEDEF_ALL(axi_lite_a48_d32, logic [47:0], logic [31:0], logic [3:0])

  /// Inputs of the `soc_axi_lite_narrow_periph_xbar` crossbar.
  typedef enum int {
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_IN_SOC,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_NUM_INPUTS
  } soc_axi_lite_narrow_periph_xbar_inputs_e;

  /// Outputs of the `soc_axi_lite_narrow_periph_xbar` crossbar.
  typedef enum int {
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_SOC_CTRL,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_FLL_SYSTEM,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_FLL_PERIPH,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_FLL_HBM2E,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_UART,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_GPIO,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_I2C,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_CHIP_CTRL,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_TIMER,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_HBM_XBAR_CFG,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_SPIM,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_PCIE_CFG,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_HBI_WIDE_CFG,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_HBI_NARROW_CFG,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_PLIC,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_BOOTROM,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_CLINT,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_OUT_HBM_CFG,
    SOC_AXI_LITE_NARROW_PERIPH_XBAR_NUM_OUTPUTS
  } soc_axi_lite_narrow_periph_xbar_outputs_e;

  /// Configuration of the `soc_axi_lite_narrow_periph_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t SocAxiLiteNarrowPeriphXbarCfg = '{
      NoSlvPorts: SOC_AXI_LITE_NARROW_PERIPH_XBAR_NUM_INPUTS,
      NoMstPorts: SOC_AXI_LITE_NARROW_PERIPH_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 4,
      MaxMstTrans: 4,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 0,
      AxiIdUsedSlvPorts: 0,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 32,
      NoAddrRules: 18
  };

  // AXI plugs of the `soc_axi_lite_narrow_periph_xbar` crossbar.

  typedef axi_lite_a48_d32_req_t soc_axi_lite_narrow_periph_xbar_in_req_t;
  typedef axi_lite_a48_d32_req_t soc_axi_lite_narrow_periph_xbar_out_req_t;
  typedef axi_lite_a48_d32_rsp_t soc_axi_lite_narrow_periph_xbar_in_rsp_t;
  typedef axi_lite_a48_d32_rsp_t soc_axi_lite_narrow_periph_xbar_out_rsp_t;
  typedef axi_lite_a48_d32_aw_chan_t soc_axi_lite_narrow_periph_xbar_in_aw_chan_t;
  typedef axi_lite_a48_d32_aw_chan_t soc_axi_lite_narrow_periph_xbar_out_aw_chan_t;
  typedef axi_lite_a48_d32_w_chan_t soc_axi_lite_narrow_periph_xbar_in_w_chan_t;
  typedef axi_lite_a48_d32_w_chan_t soc_axi_lite_narrow_periph_xbar_out_w_chan_t;
  typedef axi_lite_a48_d32_b_chan_t soc_axi_lite_narrow_periph_xbar_in_b_chan_t;
  typedef axi_lite_a48_d32_b_chan_t soc_axi_lite_narrow_periph_xbar_out_b_chan_t;
  typedef axi_lite_a48_d32_ar_chan_t soc_axi_lite_narrow_periph_xbar_in_ar_chan_t;
  typedef axi_lite_a48_d32_ar_chan_t soc_axi_lite_narrow_periph_xbar_out_ar_chan_t;
  typedef axi_lite_a48_d32_r_chan_t soc_axi_lite_narrow_periph_xbar_in_r_chan_t;
  typedef axi_lite_a48_d32_r_chan_t soc_axi_lite_narrow_periph_xbar_out_r_chan_t;

  /// Inputs of the `quadrant_pre_xbar_0` crossbar.
  typedef enum int {
    QUADRANT_PRE_XBAR_0_IN_QUADRANT,
    QUADRANT_PRE_XBAR_0_NUM_INPUTS
  } quadrant_pre_xbar_0_inputs_e;

  /// Outputs of the `quadrant_pre_xbar_0` crossbar.
  typedef enum int {
    QUADRANT_PRE_XBAR_0_OUT_QUADRANT_INTER_XBAR,
    QUADRANT_PRE_XBAR_0_OUT_HBM_XBAR,
    QUADRANT_PRE_XBAR_0_NUM_OUTPUTS
  } quadrant_pre_xbar_0_outputs_e;

  /// Configuration of the `quadrant_pre_xbar_0` crossbar.
  localparam axi_pkg::xbar_cfg_t QuadrantPreXbar0Cfg = '{
      NoSlvPorts: QUADRANT_PRE_XBAR_0_NUM_INPUTS,
      NoMstPorts: QUADRANT_PRE_XBAR_0_NUM_OUTPUTS,
      MaxSlvTrans: 64,
      MaxMstTrans: 64,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 5
  };

  // AXI bus with 48 bit address, 512 bit data, 4 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i4_u0, logic [47:0], logic [3:0], logic [511:0], logic [63:0],
                   logic [0:0])

  typedef axi_a48_d512_i4_u0_req_t quadrant_pre_xbar_0_in_req_t;
  typedef axi_a48_d512_i4_u0_req_t quadrant_pre_xbar_0_out_req_t;
  typedef axi_a48_d512_i4_u0_resp_t quadrant_pre_xbar_0_in_resp_t;
  typedef axi_a48_d512_i4_u0_resp_t quadrant_pre_xbar_0_out_resp_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t quadrant_pre_xbar_0_in_aw_chan_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t quadrant_pre_xbar_0_out_aw_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t quadrant_pre_xbar_0_in_w_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t quadrant_pre_xbar_0_out_w_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t quadrant_pre_xbar_0_in_b_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t quadrant_pre_xbar_0_out_b_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t quadrant_pre_xbar_0_in_ar_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t quadrant_pre_xbar_0_out_ar_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t quadrant_pre_xbar_0_in_r_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t quadrant_pre_xbar_0_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_PRE_XBAR_0_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_PRE_XBAR_0_IW_OUT = 4;

  /// Inputs of the `quadrant_inter_xbar` crossbar.
  typedef enum int {
    QUADRANT_INTER_XBAR_IN_WIDE_XBAR,
    QUADRANT_INTER_XBAR_IN_QUADRANT_0,
    QUADRANT_INTER_XBAR_NUM_INPUTS
  } quadrant_inter_xbar_inputs_e;

  /// Outputs of the `quadrant_inter_xbar` crossbar.
  typedef enum int {
    QUADRANT_INTER_XBAR_OUT_WIDE_XBAR,
    QUADRANT_INTER_XBAR_OUT_QUADRANT_0,
    QUADRANT_INTER_XBAR_NUM_OUTPUTS
  } quadrant_inter_xbar_outputs_e;

  /// Configuration of the `quadrant_inter_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t QuadrantInterXbarCfg = '{
      NoSlvPorts: QUADRANT_INTER_XBAR_NUM_INPUTS,
      NoMstPorts: QUADRANT_INTER_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 64,
      MaxMstTrans: 64,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 3
  };

  // AXI bus with 48 bit address, 512 bit data, 5 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i5_u0, logic [47:0], logic [4:0], logic [511:0], logic [63:0],
                   logic [0:0])

  typedef axi_a48_d512_i4_u0_req_t quadrant_inter_xbar_in_req_t;
  typedef axi_a48_d512_i5_u0_req_t quadrant_inter_xbar_out_req_t;
  typedef axi_a48_d512_i4_u0_resp_t quadrant_inter_xbar_in_resp_t;
  typedef axi_a48_d512_i5_u0_resp_t quadrant_inter_xbar_out_resp_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t quadrant_inter_xbar_in_aw_chan_t;
  typedef axi_a48_d512_i5_u0_aw_chan_t quadrant_inter_xbar_out_aw_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t quadrant_inter_xbar_in_w_chan_t;
  typedef axi_a48_d512_i5_u0_w_chan_t quadrant_inter_xbar_out_w_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t quadrant_inter_xbar_in_b_chan_t;
  typedef axi_a48_d512_i5_u0_b_chan_t quadrant_inter_xbar_out_b_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t quadrant_inter_xbar_in_ar_chan_t;
  typedef axi_a48_d512_i5_u0_ar_chan_t quadrant_inter_xbar_out_ar_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t quadrant_inter_xbar_in_r_chan_t;
  typedef axi_a48_d512_i5_u0_r_chan_t quadrant_inter_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_INTER_XBAR_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_INTER_XBAR_IW_OUT = 5;

  /// Inputs of the `hbm_xbar` crossbar.
  typedef enum int {
    HBM_XBAR_IN_QUADRANT_0,
    HBM_XBAR_IN_WIDE_XBAR,
    HBM_XBAR_NUM_INPUTS
  } hbm_xbar_inputs_e;

  /// Outputs of the `hbm_xbar` crossbar.
  typedef enum int {
    HBM_XBAR_OUT_HBM_0,
    HBM_XBAR_OUT_HBM_1,
    HBM_XBAR_OUT_HBM_2,
    HBM_XBAR_OUT_HBM_3,
    HBM_XBAR_OUT_HBM_4,
    HBM_XBAR_OUT_HBM_5,
    HBM_XBAR_OUT_HBM_6,
    HBM_XBAR_OUT_HBM_7,
    HBM_XBAR_NUM_OUTPUTS
  } hbm_xbar_outputs_e;

  /// Configuration of the `hbm_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t HbmXbarCfg = '{
      NoSlvPorts: HBM_XBAR_NUM_INPUTS,
      NoMstPorts: HBM_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 128,
      MaxMstTrans: 128,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 10
  };

  typedef axi_a48_d512_i4_u0_req_t hbm_xbar_in_req_t;
  typedef axi_a48_d512_i5_u0_req_t hbm_xbar_out_req_t;
  typedef axi_a48_d512_i4_u0_resp_t hbm_xbar_in_resp_t;
  typedef axi_a48_d512_i5_u0_resp_t hbm_xbar_out_resp_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t hbm_xbar_in_aw_chan_t;
  typedef axi_a48_d512_i5_u0_aw_chan_t hbm_xbar_out_aw_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t hbm_xbar_in_w_chan_t;
  typedef axi_a48_d512_i5_u0_w_chan_t hbm_xbar_out_w_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t hbm_xbar_in_b_chan_t;
  typedef axi_a48_d512_i5_u0_b_chan_t hbm_xbar_out_b_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t hbm_xbar_in_ar_chan_t;
  typedef axi_a48_d512_i5_u0_ar_chan_t hbm_xbar_out_ar_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t hbm_xbar_in_r_chan_t;
  typedef axi_a48_d512_i5_u0_r_chan_t hbm_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int HBM_XBAR_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int HBM_XBAR_IW_OUT = 5;

  /// Inputs of the `soc_wide_xbar` crossbar.
  typedef enum int {
    SOC_WIDE_XBAR_IN_HBI,
    SOC_WIDE_XBAR_IN_QUADRANT_INTER_XBAR,
    SOC_WIDE_XBAR_IN_SOC_NARROW,
    SOC_WIDE_XBAR_IN_SYS_IDMA_MST,
    SOC_WIDE_XBAR_NUM_INPUTS
  } soc_wide_xbar_inputs_e;

  /// Outputs of the `soc_wide_xbar` crossbar.
  typedef enum int {
    SOC_WIDE_XBAR_OUT_HBI,
    SOC_WIDE_XBAR_OUT_HBM_XBAR,
    SOC_WIDE_XBAR_OUT_QUADRANT_INTER_XBAR,
    SOC_WIDE_XBAR_OUT_SOC_NARROW,
    SOC_WIDE_XBAR_OUT_SPM_WIDE,
    SOC_WIDE_XBAR_OUT_WIDE_ZERO_MEM,
    SOC_WIDE_XBAR_NUM_OUTPUTS
  } soc_wide_xbar_outputs_e;

  /// Configuration of the `soc_wide_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t SocWideXbarCfg = '{
      NoSlvPorts: SOC_WIDE_XBAR_NUM_INPUTS,
      NoMstPorts: SOC_WIDE_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 64,
      MaxMstTrans: 64,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 8
  };

  // AXI bus with 48 bit address, 512 bit data, 6 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i6_u0, logic [47:0], logic [5:0], logic [511:0], logic [63:0],
                   logic [0:0])

  typedef axi_a48_d512_i4_u0_req_t soc_wide_xbar_in_req_t;
  typedef axi_a48_d512_i6_u0_req_t soc_wide_xbar_out_req_t;
  typedef axi_a48_d512_i4_u0_resp_t soc_wide_xbar_in_resp_t;
  typedef axi_a48_d512_i6_u0_resp_t soc_wide_xbar_out_resp_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t soc_wide_xbar_in_aw_chan_t;
  typedef axi_a48_d512_i6_u0_aw_chan_t soc_wide_xbar_out_aw_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t soc_wide_xbar_in_w_chan_t;
  typedef axi_a48_d512_i6_u0_w_chan_t soc_wide_xbar_out_w_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t soc_wide_xbar_in_b_chan_t;
  typedef axi_a48_d512_i6_u0_b_chan_t soc_wide_xbar_out_b_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t soc_wide_xbar_in_ar_chan_t;
  typedef axi_a48_d512_i6_u0_ar_chan_t soc_wide_xbar_out_ar_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t soc_wide_xbar_in_r_chan_t;
  typedef axi_a48_d512_i6_u0_r_chan_t soc_wide_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int SOC_WIDE_XBAR_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int SOC_WIDE_XBAR_IW_OUT = 6;

  /// Inputs of the `soc_narrow_xbar` crossbar.
  typedef enum int {
    SOC_NARROW_XBAR_IN_S1_QUADRANT_0,
    SOC_NARROW_XBAR_IN_CVA6,
    SOC_NARROW_XBAR_IN_SOC_WIDE,
    SOC_NARROW_XBAR_IN_PERIPH,
    SOC_NARROW_XBAR_IN_PCIE,
    SOC_NARROW_XBAR_IN_HBI,
    SOC_NARROW_XBAR_NUM_INPUTS
  } soc_narrow_xbar_inputs_e;

  /// Outputs of the `soc_narrow_xbar` crossbar.
  typedef enum int {
    SOC_NARROW_XBAR_OUT_S1_QUADRANT_0,
    SOC_NARROW_XBAR_OUT_SOC_WIDE,
    SOC_NARROW_XBAR_OUT_HBI,
    SOC_NARROW_XBAR_OUT_PERIPH,
    SOC_NARROW_XBAR_OUT_SPM_NARROW,
    SOC_NARROW_XBAR_OUT_SYS_IDMA_CFG,
    SOC_NARROW_XBAR_OUT_AXI_LITE_NARROW_PERIPH,
    SOC_NARROW_XBAR_OUT_PCIE,
    SOC_NARROW_XBAR_NUM_OUTPUTS
  } soc_narrow_xbar_outputs_e;

  /// Configuration of the `soc_narrow_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t SocNarrowXbarCfg = '{
      NoSlvPorts: SOC_NARROW_XBAR_NUM_INPUTS,
      NoMstPorts: SOC_NARROW_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 32,
      MaxMstTrans: 32,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 10
  };

  // AXI bus with 48 bit address, 64 bit data, 4 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i4_u5, logic [47:0], logic [3:0], logic [63:0], logic [7:0],
                   logic [4:0])

  // AXI bus with 48 bit address, 64 bit data, 7 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i7_u5, logic [47:0], logic [6:0], logic [63:0], logic [7:0],
                   logic [4:0])

  typedef axi_a48_d64_i4_u5_req_t soc_narrow_xbar_in_req_t;
  typedef axi_a48_d64_i7_u5_req_t soc_narrow_xbar_out_req_t;
  typedef axi_a48_d64_i4_u5_resp_t soc_narrow_xbar_in_resp_t;
  typedef axi_a48_d64_i7_u5_resp_t soc_narrow_xbar_out_resp_t;
  typedef axi_a48_d64_i4_u5_aw_chan_t soc_narrow_xbar_in_aw_chan_t;
  typedef axi_a48_d64_i7_u5_aw_chan_t soc_narrow_xbar_out_aw_chan_t;
  typedef axi_a48_d64_i4_u5_w_chan_t soc_narrow_xbar_in_w_chan_t;
  typedef axi_a48_d64_i7_u5_w_chan_t soc_narrow_xbar_out_w_chan_t;
  typedef axi_a48_d64_i4_u5_b_chan_t soc_narrow_xbar_in_b_chan_t;
  typedef axi_a48_d64_i7_u5_b_chan_t soc_narrow_xbar_out_b_chan_t;
  typedef axi_a48_d64_i4_u5_ar_chan_t soc_narrow_xbar_in_ar_chan_t;
  typedef axi_a48_d64_i7_u5_ar_chan_t soc_narrow_xbar_out_ar_chan_t;
  typedef axi_a48_d64_i4_u5_r_chan_t soc_narrow_xbar_in_r_chan_t;
  typedef axi_a48_d64_i7_u5_r_chan_t soc_narrow_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int SOC_NARROW_XBAR_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int SOC_NARROW_XBAR_IW_OUT = 7;

  /// Inputs of the `quadrant_s1_ctrl_soc_to_quad_xbar` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_IN_IN,
    QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_NUM_INPUTS
  } quadrant_s1_ctrl_soc_to_quad_xbar_inputs_e;

  /// Outputs of the `quadrant_s1_ctrl_soc_to_quad_xbar` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_OUT_OUT,
    QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_OUT_INTERNAL,
    QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_NUM_OUTPUTS
  } quadrant_s1_ctrl_soc_to_quad_xbar_outputs_e;

  /// Configuration of the `quadrant_s1_ctrl_soc_to_quad_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t QuadrantS1CtrlSocToQuadXbarCfg = '{
      NoSlvPorts: QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_NUM_INPUTS,
      NoMstPorts: QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 32,
      MaxMstTrans: 32,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_SLV_PORTS,
      AxiIdWidthSlvPorts: 7,
      AxiIdUsedSlvPorts: 7,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 1
  };

  typedef axi_a48_d64_i7_u5_req_t quadrant_s1_ctrl_soc_to_quad_xbar_in_req_t;
  typedef axi_a48_d64_i7_u5_req_t quadrant_s1_ctrl_soc_to_quad_xbar_out_req_t;
  typedef axi_a48_d64_i7_u5_resp_t quadrant_s1_ctrl_soc_to_quad_xbar_in_resp_t;
  typedef axi_a48_d64_i7_u5_resp_t quadrant_s1_ctrl_soc_to_quad_xbar_out_resp_t;
  typedef axi_a48_d64_i7_u5_aw_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_in_aw_chan_t;
  typedef axi_a48_d64_i7_u5_aw_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_out_aw_chan_t;
  typedef axi_a48_d64_i7_u5_w_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_in_w_chan_t;
  typedef axi_a48_d64_i7_u5_w_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_out_w_chan_t;
  typedef axi_a48_d64_i7_u5_b_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_in_b_chan_t;
  typedef axi_a48_d64_i7_u5_b_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_out_b_chan_t;
  typedef axi_a48_d64_i7_u5_ar_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_in_ar_chan_t;
  typedef axi_a48_d64_i7_u5_ar_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_out_ar_chan_t;
  typedef axi_a48_d64_i7_u5_r_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_in_r_chan_t;
  typedef axi_a48_d64_i7_u5_r_chan_t quadrant_s1_ctrl_soc_to_quad_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_IW_IN = 7;
  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_S1_CTRL_SOC_TO_QUAD_XBAR_IW_OUT = 7;

  /// Inputs of the `quadrant_s1_ctrl_quad_to_soc_xbar` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_IN_IN,
    QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_NUM_INPUTS
  } quadrant_s1_ctrl_quad_to_soc_xbar_inputs_e;

  /// Outputs of the `quadrant_s1_ctrl_quad_to_soc_xbar` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_OUT_OUT,
    QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_OUT_INTERNAL,
    QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_NUM_OUTPUTS
  } quadrant_s1_ctrl_quad_to_soc_xbar_outputs_e;

  /// Configuration of the `quadrant_s1_ctrl_quad_to_soc_xbar` crossbar.
  localparam axi_pkg::xbar_cfg_t QuadrantS1CtrlQuadToSocXbarCfg = '{
      NoSlvPorts: QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_NUM_INPUTS,
      NoMstPorts: QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_NUM_OUTPUTS,
      MaxSlvTrans: 32,
      MaxMstTrans: 32,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_MST_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 1
  };

  typedef axi_a48_d64_i4_u5_req_t quadrant_s1_ctrl_quad_to_soc_xbar_in_req_t;
  typedef axi_a48_d64_i4_u5_req_t quadrant_s1_ctrl_quad_to_soc_xbar_out_req_t;
  typedef axi_a48_d64_i4_u5_resp_t quadrant_s1_ctrl_quad_to_soc_xbar_in_resp_t;
  typedef axi_a48_d64_i4_u5_resp_t quadrant_s1_ctrl_quad_to_soc_xbar_out_resp_t;
  typedef axi_a48_d64_i4_u5_aw_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_in_aw_chan_t;
  typedef axi_a48_d64_i4_u5_aw_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_out_aw_chan_t;
  typedef axi_a48_d64_i4_u5_w_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_in_w_chan_t;
  typedef axi_a48_d64_i4_u5_w_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_out_w_chan_t;
  typedef axi_a48_d64_i4_u5_b_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_in_b_chan_t;
  typedef axi_a48_d64_i4_u5_b_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_out_b_chan_t;
  typedef axi_a48_d64_i4_u5_ar_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_in_ar_chan_t;
  typedef axi_a48_d64_i4_u5_ar_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_out_ar_chan_t;
  typedef axi_a48_d64_i4_u5_r_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_in_r_chan_t;
  typedef axi_a48_d64_i4_u5_r_chan_t quadrant_s1_ctrl_quad_to_soc_xbar_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int QUADRANT_S1_CTRL_QUAD_TO_SOC_XBAR_IW_OUT = 4;

  /// Inputs of the `quadrant_s1_ctrl_mux` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_MUX_IN_SOC,
    QUADRANT_S1_CTRL_MUX_IN_QUAD,
    QUADRANT_S1_CTRL_MUX_NUM_INPUTS
  } quadrant_s1_ctrl_mux_inputs_e;

  /// Outputs of the `quadrant_s1_ctrl_mux` crossbar.
  typedef enum int {
    QUADRANT_S1_CTRL_MUX_OUT_OUT,
    QUADRANT_S1_CTRL_MUX_NUM_OUTPUTS
  } quadrant_s1_ctrl_mux_outputs_e;

  /// Configuration of the `quadrant_s1_ctrl_mux` crossbar.
  localparam axi_pkg::xbar_cfg_t QuadrantS1CtrlMuxCfg = '{
      NoSlvPorts: QUADRANT_S1_CTRL_MUX_NUM_INPUTS,
      NoMstPorts: QUADRANT_S1_CTRL_MUX_NUM_OUTPUTS,
      MaxSlvTrans: 32,
      MaxMstTrans: 32,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 0,
      AxiIdUsedSlvPorts: 0,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 32,
      NoAddrRules: 1
  };

  // AXI plugs of the `quadrant_s1_ctrl_mux` crossbar.

  typedef axi_lite_a48_d32_req_t quadrant_s1_ctrl_mux_in_req_t;
  typedef axi_lite_a48_d32_req_t quadrant_s1_ctrl_mux_out_req_t;
  typedef axi_lite_a48_d32_rsp_t quadrant_s1_ctrl_mux_in_rsp_t;
  typedef axi_lite_a48_d32_rsp_t quadrant_s1_ctrl_mux_out_rsp_t;
  typedef axi_lite_a48_d32_aw_chan_t quadrant_s1_ctrl_mux_in_aw_chan_t;
  typedef axi_lite_a48_d32_aw_chan_t quadrant_s1_ctrl_mux_out_aw_chan_t;
  typedef axi_lite_a48_d32_w_chan_t quadrant_s1_ctrl_mux_in_w_chan_t;
  typedef axi_lite_a48_d32_w_chan_t quadrant_s1_ctrl_mux_out_w_chan_t;
  typedef axi_lite_a48_d32_b_chan_t quadrant_s1_ctrl_mux_in_b_chan_t;
  typedef axi_lite_a48_d32_b_chan_t quadrant_s1_ctrl_mux_out_b_chan_t;
  typedef axi_lite_a48_d32_ar_chan_t quadrant_s1_ctrl_mux_in_ar_chan_t;
  typedef axi_lite_a48_d32_ar_chan_t quadrant_s1_ctrl_mux_out_ar_chan_t;
  typedef axi_lite_a48_d32_r_chan_t quadrant_s1_ctrl_mux_in_r_chan_t;
  typedef axi_lite_a48_d32_r_chan_t quadrant_s1_ctrl_mux_out_r_chan_t;

  /// Inputs of the `wide_xbar_quadrant_s1` crossbar.
  typedef enum int {
    WIDE_XBAR_QUADRANT_S1_IN_TOP,
    WIDE_XBAR_QUADRANT_S1_IN_CLUSTER_0,
    WIDE_XBAR_QUADRANT_S1_NUM_INPUTS
  } wide_xbar_quadrant_s1_inputs_e;

  /// Outputs of the `wide_xbar_quadrant_s1` crossbar.
  typedef enum int {
    WIDE_XBAR_QUADRANT_S1_OUT_TOP,
    WIDE_XBAR_QUADRANT_S1_OUT_CLUSTER_0,
    WIDE_XBAR_QUADRANT_S1_NUM_OUTPUTS
  } wide_xbar_quadrant_s1_outputs_e;

  /// Configuration of the `wide_xbar_quadrant_s1` crossbar.
  localparam axi_pkg::xbar_cfg_t WideXbarQuadrantS1Cfg = '{
      NoSlvPorts: WIDE_XBAR_QUADRANT_S1_NUM_INPUTS,
      NoMstPorts: WIDE_XBAR_QUADRANT_S1_NUM_OUTPUTS,
      MaxSlvTrans: 32,
      MaxMstTrans: 32,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 3,
      AxiIdUsedSlvPorts: 3,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 1
  };

  // AXI bus with 48 bit address, 512 bit data, 3 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i3_u0, logic [47:0], logic [2:0], logic [511:0], logic [63:0],
                   logic [0:0])

  typedef axi_a48_d512_i3_u0_req_t wide_xbar_quadrant_s1_in_req_t;
  typedef axi_a48_d512_i4_u0_req_t wide_xbar_quadrant_s1_out_req_t;
  typedef axi_a48_d512_i3_u0_resp_t wide_xbar_quadrant_s1_in_resp_t;
  typedef axi_a48_d512_i4_u0_resp_t wide_xbar_quadrant_s1_out_resp_t;
  typedef axi_a48_d512_i3_u0_aw_chan_t wide_xbar_quadrant_s1_in_aw_chan_t;
  typedef axi_a48_d512_i4_u0_aw_chan_t wide_xbar_quadrant_s1_out_aw_chan_t;
  typedef axi_a48_d512_i3_u0_w_chan_t wide_xbar_quadrant_s1_in_w_chan_t;
  typedef axi_a48_d512_i4_u0_w_chan_t wide_xbar_quadrant_s1_out_w_chan_t;
  typedef axi_a48_d512_i3_u0_b_chan_t wide_xbar_quadrant_s1_in_b_chan_t;
  typedef axi_a48_d512_i4_u0_b_chan_t wide_xbar_quadrant_s1_out_b_chan_t;
  typedef axi_a48_d512_i3_u0_ar_chan_t wide_xbar_quadrant_s1_in_ar_chan_t;
  typedef axi_a48_d512_i4_u0_ar_chan_t wide_xbar_quadrant_s1_out_ar_chan_t;
  typedef axi_a48_d512_i3_u0_r_chan_t wide_xbar_quadrant_s1_in_r_chan_t;
  typedef axi_a48_d512_i4_u0_r_chan_t wide_xbar_quadrant_s1_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int WIDE_XBAR_QUADRANT_S1_IW_IN = 3;
  // verilog_lint: waive parameter-name-style
  localparam int WIDE_XBAR_QUADRANT_S1_IW_OUT = 4;

  /// Inputs of the `narrow_xbar_quadrant_s1` crossbar.
  typedef enum int {
    NARROW_XBAR_QUADRANT_S1_IN_TOP,
    NARROW_XBAR_QUADRANT_S1_IN_CLUSTER_0,
    NARROW_XBAR_QUADRANT_S1_NUM_INPUTS
  } narrow_xbar_quadrant_s1_inputs_e;

  /// Outputs of the `narrow_xbar_quadrant_s1` crossbar.
  typedef enum int {
    NARROW_XBAR_QUADRANT_S1_OUT_TOP,
    NARROW_XBAR_QUADRANT_S1_OUT_CLUSTER_0,
    NARROW_XBAR_QUADRANT_S1_NUM_OUTPUTS
  } narrow_xbar_quadrant_s1_outputs_e;

  /// Configuration of the `narrow_xbar_quadrant_s1` crossbar.
  localparam axi_pkg::xbar_cfg_t NarrowXbarQuadrantS1Cfg = '{
      NoSlvPorts: NARROW_XBAR_QUADRANT_S1_NUM_INPUTS,
      NoMstPorts: NARROW_XBAR_QUADRANT_S1_NUM_OUTPUTS,
      MaxSlvTrans: 8,
      MaxMstTrans: 8,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 1
  };

  // AXI bus with 48 bit address, 64 bit data, 5 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i5_u5, logic [47:0], logic [4:0], logic [63:0], logic [7:0],
                   logic [4:0])

  typedef axi_a48_d64_i4_u5_req_t narrow_xbar_quadrant_s1_in_req_t;
  typedef axi_a48_d64_i5_u5_req_t narrow_xbar_quadrant_s1_out_req_t;
  typedef axi_a48_d64_i4_u5_resp_t narrow_xbar_quadrant_s1_in_resp_t;
  typedef axi_a48_d64_i5_u5_resp_t narrow_xbar_quadrant_s1_out_resp_t;
  typedef axi_a48_d64_i4_u5_aw_chan_t narrow_xbar_quadrant_s1_in_aw_chan_t;
  typedef axi_a48_d64_i5_u5_aw_chan_t narrow_xbar_quadrant_s1_out_aw_chan_t;
  typedef axi_a48_d64_i4_u5_w_chan_t narrow_xbar_quadrant_s1_in_w_chan_t;
  typedef axi_a48_d64_i5_u5_w_chan_t narrow_xbar_quadrant_s1_out_w_chan_t;
  typedef axi_a48_d64_i4_u5_b_chan_t narrow_xbar_quadrant_s1_in_b_chan_t;
  typedef axi_a48_d64_i5_u5_b_chan_t narrow_xbar_quadrant_s1_out_b_chan_t;
  typedef axi_a48_d64_i4_u5_ar_chan_t narrow_xbar_quadrant_s1_in_ar_chan_t;
  typedef axi_a48_d64_i5_u5_ar_chan_t narrow_xbar_quadrant_s1_out_ar_chan_t;
  typedef axi_a48_d64_i4_u5_r_chan_t narrow_xbar_quadrant_s1_in_r_chan_t;
  typedef axi_a48_d64_i5_u5_r_chan_t narrow_xbar_quadrant_s1_out_r_chan_t;

  // verilog_lint: waive parameter-name-style
  localparam int NARROW_XBAR_QUADRANT_S1_IW_IN = 4;
  // verilog_lint: waive parameter-name-style
  localparam int NARROW_XBAR_QUADRANT_S1_IW_OUT = 5;

  // APB bus with 48 bit address, 32 bit data.
  `APB_TYPEDEF_REQ_T(apb_a48_d32_req_t, logic [47:0], logic [31:0], logic [3:0])
  `APB_TYPEDEF_RESP_T(apb_a48_d32_rsp_t, logic [31:0])

  // AXI bus with 48 bit address, 32 bit data, 7 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d32_i7_u5, logic [47:0], logic [6:0], logic [31:0], logic [3:0],
                   logic [4:0])

  // Register bus with 48 bit address and 64 bit data.
  `REG_BUS_TYPEDEF_ALL(reg_a48_d64, logic [47:0], logic [63:0], logic [7:0])

  // AXI bus with 48 bit address, 64 bit data, 4 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i4_u0, logic [47:0], logic [3:0], logic [63:0], logic [7:0],
                   logic [0:0])

  // AXI bus with 48 bit address, 512 bit data, 4 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i4_u5, logic [47:0], logic [3:0], logic [511:0], logic [63:0],
                   logic [4:0])

  // AXI bus with 48 bit address, 64 bit data, 1 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i1_u5, logic [47:0], logic [0:0], logic [63:0], logic [7:0],
                   logic [4:0])

  // AXI bus with 48 bit address, 32 bit data, 1 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d32_i1_u5, logic [47:0], logic [0:0], logic [31:0], logic [3:0],
                   logic [4:0])

  // AXI bus with 48 bit address, 64 bit data, 2 bit IDs, and 5 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d64_i2_u5, logic [47:0], logic [1:0], logic [63:0], logic [7:0],
                   logic [4:0])

  // AXI bus with 48 bit address, 512 bit data, 1 bit IDs, and 0 bit user data.
  `AXI_TYPEDEF_ALL(axi_a48_d512_i1_u0, logic [47:0], logic [0:0], logic [511:0], logic [63:0],
                   logic [0:0])


endpackage
// verilog_lint: waive-off line-length
