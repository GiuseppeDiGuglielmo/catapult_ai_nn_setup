# Establish the location of this script and use it to reference all
# other files in this example
set sfd [file dirname [info script]]

# Reset the options to the factory defaults
options defaults

project new

# Enable the HLS4ML flow
flow package require /HLS4ML

# Create a Python Virtual Environment populated with HLS4ML
flow run /HLS4ML/create_venv

# Run HLS4ML to generate the C++ and Catapult project TCL
flow run /HLS4ML/gen_hls4ml $sfd/model_asic.py --reuse_factor 1

# Now 'dofile' the project
if { [file exists [pwd]/my-Catapult-test_asic1] } {
  set ::argv "csim=True Synth=True SCVerify=True RTLSynth=False BuildBUP=True"
  set_working_dir [pwd]/my-Catapult-test_asic1
  dofile build_prj.tcl
  go switching
  flow run /PowerAnalysis/report_pre_pwropt_Verilog
  flow run /HLS4ML/collect_reports
} else {
  logfile message "HLS4ML failed to generate the model in directory '[pwd]/my-Catapult-test_asic1'\n" error
}

