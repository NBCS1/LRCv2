args = getArgument();

// Split arguments by semicolon (;)
tokens = split(args, ";");

exp_nro = parseInt(tokens[0]);
files = tokens[1];
XY_registration = tokens[2];
projection_type_xy = tokens[3];
time_xy = parseInt(tokens[4]);
max_xy = parseInt(tokens[5]);
reference_xy = tokens[6];
crop_output = tokens[7];
z_registration = tokens[8];
projection_type_z = tokens[9];
reslice_mode = tokens[10];
time_z = parseInt(tokens[11]);
reference_z = tokens[12];
extend_stack_to_fit = tokens[13];
ram_conservative_mode = tokens[14];
max_z = parseInt(tokens[15]);
date=tokens[17]
// Combine them into a single string for the macro
params = "exp_nro=" + exp_nro + " " +
         "files=[" + files + "] " +
         "XY_registration=" + XY_registration + " " +
         "projection_type_xy=[" + projection_type_xy + "] " +
         "time_xy=" + time_xy + " " +
         "max_xy=" + max_xy + " " +
         "reference_xy=[" + reference_xy + "] " +
         "crop_output=" + crop_output + " " +
         "z_registration=" + z_registration + " " +
         "projection_type_z=[" + projection_type_z + "] " +
         "reslice_mode=[" + reslice_mode + "] " +
         "time_z=" + time_z + " " +
         "reference_z=[" + reference_z + "] " +
         "extend_stack_to_fit=" + extend_stack_to_fit + " " +
         "ram_conservative_mode=" + ram_conservative_mode + " " +
         "max_z=" + max_z;

// Run the command
run("time estimate+apply", params);
// Combine them into a single string

files2 = tokens[16];
pathArrayC1 = split(files2, ",");
pathArrayC2 = split(files, ",");

for (i = 0; i < pathArrayC2.length; i++) {
	lastSlash = lastIndexOf(pathArrayC2[i], "/"); // Use "\\" on Windows
	// Extract the folder path (everything before the last slash)
	workingFolder=substring(pathArrayC2[i], 0, lastSlash);
	filename =substring(pathArrayC2[i], lastSlash + 1);
	filename=replace(filename, ".tif", "");
	settings_file_path =workingFolder+"//"+filename+"_"+date+"-001//"+filename+"_settings.csv";
	output_folder=workingFolder+"//"+filename+"_"+date+"-001//";
	params2 ="files=[" + pathArrayC1[i] + "] " +
         "settings_file_path=[" + settings_file_path + "] " +
         "results_path=[" + output_folder + "] ";

         run("time apply", params2);
}
logFile = getInfo("log");
File.saveString(logFile, workingFolder+"//"+filename+"_"+date+"-001//"+filename+"_IJLogFile.txt");
run("Quit");






















