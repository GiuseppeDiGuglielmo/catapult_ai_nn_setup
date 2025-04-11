// pragma dofile( verbose : none )

# Establish the location of this script and use it to reference all
# other files in this example
set sfd [file dirname [info script]]

##
## default workers is 3
##
## set env(CCS_WORKERS) 3
##
## default is to run workers on the current host
##
## set env(CCS_REMOTE_COMMAND) ...
##
## tokens will be substituted into the ... string
##   %HOSTNAME%           - current hostname
##   %PID%                - current process id
##   %PROJECT_HOME%       - parent directory project directory
##   %COUNT%              - number of workers - if omitted command will be called COUNT times
##   %COMMAND%            - the Catapult invocation command line
##   %ENV:NAME%           - substitute the environmental variable value, if the value is undefined an empty string will be used
##   %ENV:NAME:-DEFAULT%  - substitute the environmental variable value, if the value is undefined the DEFAULT string will be used
##

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
  dofile build_prj_bup.tcl
  go switching
  flow run /PowerAnalysis/report_pre_pwropt_Verilog
  flow run /HLS4ML/collect_reports
} else {
  logfile message "HLS4ML failed to generate the model in directory '[pwd]/my-Catapult-test_asic1'\n" error
}

