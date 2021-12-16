// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Register Package auto-generated by `reggen` containing data structure

package occamy_quadrant_s1_reg_pkg;

  // Address widths within the block
  parameter int BlockAw = 9;

  ////////////////////////////
  // Typedefs for registers //
  ////////////////////////////

  typedef struct packed {
    logic        q;
  } occamy_quadrant_s1_reg2hw_clk_ena_reg_t;

  typedef struct packed {
    logic        q;
  } occamy_quadrant_s1_reg2hw_reset_n_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } narrow_in;
    struct packed {
      logic        q;
    } narrow_out;
    struct packed {
      logic        q;
    } wide_in;
    struct packed {
      logic        q;
    } wide_out;
    struct packed {
      logic        q;
    } hbi_out;
  } occamy_quadrant_s1_reg2hw_isolate_reg_t;

  typedef struct packed {
    logic        q;
  } occamy_quadrant_s1_reg2hw_ro_cache_enable_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_low_0_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_high_0_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_low_0_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_high_0_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_low_1_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_high_1_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_low_1_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_high_1_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_low_2_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_high_2_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_low_2_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_high_2_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_low_3_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_start_addr_high_3_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_low_3_reg_t;

  typedef struct packed {
    logic [15:0] q;
  } occamy_quadrant_s1_reg2hw_ro_end_addr_high_3_reg_t;

  typedef struct packed {
    struct packed {
      logic        d;
    } narrow_in;
    struct packed {
      logic        d;
    } narrow_out;
    struct packed {
      logic        d;
    } wide_in;
    struct packed {
      logic        d;
    } wide_out;
    struct packed {
      logic        d;
    } hbi_out;
  } occamy_quadrant_s1_hw2reg_isolated_reg_t;

  // Register -> HW type
  typedef struct packed {
    occamy_quadrant_s1_reg2hw_clk_ena_reg_t clk_ena; // [391:391]
    occamy_quadrant_s1_reg2hw_reset_n_reg_t reset_n; // [390:390]
    occamy_quadrant_s1_reg2hw_isolate_reg_t isolate; // [389:385]
    occamy_quadrant_s1_reg2hw_ro_cache_enable_reg_t ro_cache_enable; // [384:384]
    occamy_quadrant_s1_reg2hw_ro_start_addr_low_0_reg_t ro_start_addr_low_0; // [383:352]
    occamy_quadrant_s1_reg2hw_ro_start_addr_high_0_reg_t ro_start_addr_high_0; // [351:336]
    occamy_quadrant_s1_reg2hw_ro_end_addr_low_0_reg_t ro_end_addr_low_0; // [335:304]
    occamy_quadrant_s1_reg2hw_ro_end_addr_high_0_reg_t ro_end_addr_high_0; // [303:288]
    occamy_quadrant_s1_reg2hw_ro_start_addr_low_1_reg_t ro_start_addr_low_1; // [287:256]
    occamy_quadrant_s1_reg2hw_ro_start_addr_high_1_reg_t ro_start_addr_high_1; // [255:240]
    occamy_quadrant_s1_reg2hw_ro_end_addr_low_1_reg_t ro_end_addr_low_1; // [239:208]
    occamy_quadrant_s1_reg2hw_ro_end_addr_high_1_reg_t ro_end_addr_high_1; // [207:192]
    occamy_quadrant_s1_reg2hw_ro_start_addr_low_2_reg_t ro_start_addr_low_2; // [191:160]
    occamy_quadrant_s1_reg2hw_ro_start_addr_high_2_reg_t ro_start_addr_high_2; // [159:144]
    occamy_quadrant_s1_reg2hw_ro_end_addr_low_2_reg_t ro_end_addr_low_2; // [143:112]
    occamy_quadrant_s1_reg2hw_ro_end_addr_high_2_reg_t ro_end_addr_high_2; // [111:96]
    occamy_quadrant_s1_reg2hw_ro_start_addr_low_3_reg_t ro_start_addr_low_3; // [95:64]
    occamy_quadrant_s1_reg2hw_ro_start_addr_high_3_reg_t ro_start_addr_high_3; // [63:48]
    occamy_quadrant_s1_reg2hw_ro_end_addr_low_3_reg_t ro_end_addr_low_3; // [47:16]
    occamy_quadrant_s1_reg2hw_ro_end_addr_high_3_reg_t ro_end_addr_high_3; // [15:0]
  } occamy_quadrant_s1_reg2hw_t;

  // HW -> register type
  typedef struct packed {
    occamy_quadrant_s1_hw2reg_isolated_reg_t isolated; // [4:0]
  } occamy_quadrant_s1_hw2reg_t;

  // Register offsets
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_CLK_ENA_OFFSET = 9'h 0;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RESET_N_OFFSET = 9'h 4;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_ISOLATE_OFFSET = 9'h 8;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_ISOLATED_OFFSET = 9'h c;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_CACHE_ENABLE_OFFSET = 9'h 10;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_0_OFFSET = 9'h 100;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_0_OFFSET = 9'h 104;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_0_OFFSET = 9'h 108;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_0_OFFSET = 9'h 10c;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_1_OFFSET = 9'h 110;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_1_OFFSET = 9'h 114;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_1_OFFSET = 9'h 118;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_1_OFFSET = 9'h 11c;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_2_OFFSET = 9'h 120;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_2_OFFSET = 9'h 124;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_2_OFFSET = 9'h 128;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_2_OFFSET = 9'h 12c;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_3_OFFSET = 9'h 130;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_3_OFFSET = 9'h 134;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_3_OFFSET = 9'h 138;
  parameter logic [BlockAw-1:0] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_3_OFFSET = 9'h 13c;

  // Reset values for hwext registers and their fields
  parameter logic [4:0] OCCAMY_QUADRANT_S1_ISOLATED_RESVAL = 5'h 1f;
  parameter logic [0:0] OCCAMY_QUADRANT_S1_ISOLATED_NARROW_IN_RESVAL = 1'h 1;
  parameter logic [0:0] OCCAMY_QUADRANT_S1_ISOLATED_NARROW_OUT_RESVAL = 1'h 1;
  parameter logic [0:0] OCCAMY_QUADRANT_S1_ISOLATED_WIDE_IN_RESVAL = 1'h 1;
  parameter logic [0:0] OCCAMY_QUADRANT_S1_ISOLATED_WIDE_OUT_RESVAL = 1'h 1;
  parameter logic [0:0] OCCAMY_QUADRANT_S1_ISOLATED_HBI_OUT_RESVAL = 1'h 1;

  // Register index
  typedef enum int {
    OCCAMY_QUADRANT_S1_CLK_ENA,
    OCCAMY_QUADRANT_S1_RESET_N,
    OCCAMY_QUADRANT_S1_ISOLATE,
    OCCAMY_QUADRANT_S1_ISOLATED,
    OCCAMY_QUADRANT_S1_RO_CACHE_ENABLE,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_0,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_0,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_0,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_0,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_1,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_1,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_1,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_1,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_2,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_2,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_2,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_2,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_3,
    OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_3,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_3,
    OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_3
  } occamy_quadrant_s1_id_e;

  // Register width information to check illegal writes
  parameter logic [3:0] OCCAMY_QUADRANT_S1_PERMIT [21] = '{
    4'b 0001, // index[ 0] OCCAMY_QUADRANT_S1_CLK_ENA
    4'b 0001, // index[ 1] OCCAMY_QUADRANT_S1_RESET_N
    4'b 0001, // index[ 2] OCCAMY_QUADRANT_S1_ISOLATE
    4'b 0001, // index[ 3] OCCAMY_QUADRANT_S1_ISOLATED
    4'b 0001, // index[ 4] OCCAMY_QUADRANT_S1_RO_CACHE_ENABLE
    4'b 1111, // index[ 5] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_0
    4'b 0011, // index[ 6] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_0
    4'b 1111, // index[ 7] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_0
    4'b 0011, // index[ 8] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_0
    4'b 1111, // index[ 9] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_1
    4'b 0011, // index[10] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_1
    4'b 1111, // index[11] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_1
    4'b 0011, // index[12] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_1
    4'b 1111, // index[13] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_2
    4'b 0011, // index[14] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_2
    4'b 1111, // index[15] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_2
    4'b 0011, // index[16] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_2
    4'b 1111, // index[17] OCCAMY_QUADRANT_S1_RO_START_ADDR_LOW_3
    4'b 0011, // index[18] OCCAMY_QUADRANT_S1_RO_START_ADDR_HIGH_3
    4'b 1111, // index[19] OCCAMY_QUADRANT_S1_RO_END_ADDR_LOW_3
    4'b 0011  // index[20] OCCAMY_QUADRANT_S1_RO_END_ADDR_HIGH_3
  };

endpackage
