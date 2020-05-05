dir = getDirectory("Pick date to process!");

//pick out the folders and do the operations per folder
list = getFileList(dir);

for (i=0; i<list.length; i++) {
	//if this is a folder, concatenate the images inside!
	if (endsWith(list[i], "/")) 
	{
		experiment = list[i];
		list2 = getFileList(dir+experiment);
		tifname=list2[0];
		run("Image Sequence...", "open="+dir+list[i]+tifname+" sort use");

		run("8-bit");
		run("Smooth","stack");
		run("Unsharp Mask...", "radius=1 mask=0.90 stack");
		run("Auto Threshold...", "method=Default white stack");

		title = substring(experiment,0,lengthOf(experiment)-1);

		saveAs("ZIP", dir+title+".zip");
		close();
	}
}
   
