
int getAntennaAll() {

  TFile* f1 = new TFile("sim_graphs_all.root", "RECREATE");
  char *graph_name = new char[10];

  int phi=0; 
  if (phi < 0 || phi > 355 || phi % 5 != 0) return -1; 

  for(int theta = 0; theta < 94; theta += 5) {
    if (theta < 0 || theta > 180 || theta % 5 != 0) return -1; 
    std::cout << "theta " << theta << std::endl;

    char buf[512]; 
    FILE * f = fopen("./VPol_XFDTD_FullPlate.txt","r"); 
    std::vector<double> freqs; 
    std::vector<double> mags; 
    std::vector<double> swrs;
    std::vector<double> phases; 
    double freq = 0; 
    double swr = 0; 
    bool looking_for_freq = true; 
    bool looking_for_swr = true; 
    while(fgets(buf,512,f)) {
      if(looking_for_freq or looking_for_swr) {
	if(strstr(buf,"freq :")) {
	  sscanf(buf, "freq : %lg MHz", &freq); 
	  looking_for_freq = false;
	} else if(strstr(buf,"SWR :")) {
	  sscanf(buf, "SWR : %lg", &swr);
	  swrs.push_back(swr);
	  looking_for_swr = false;
	} else {
	  continue;
	}       	
      } else {
	int t,p; 
	double gdb, g, ph; 
	//now find the right theta and phi 
	int nfound = sscanf(buf,"%d %d %lf %lf %lf",&t,&p, &gdb, &g, &ph); 
	if (nfound < 5) continue; 
	if (t==theta && p ==phi ) {
	  freqs.push_back(freq/1000);
	  mags.push_back(gdb); 
	  phases.push_back(ph * TMath::DegToRad()); 
	  looking_for_freq = true; 
	  looking_for_swr  = true;
	}
      }
    }


    TGraph * gmag = new TGraph(freqs.size()); 
    gmag->SetTitle("Antenna magnitude"); 
    TGraph * gphase = new TGraph(freqs.size()); 
    gphase->SetTitle("Antenna phase"); 
    TGraph * gswr = new TGraph(swrs.size()); 
    gswr->SetTitle("SWR");
    TGraph * gpr = new TGraph(swrs.size());
    gpr->SetTitle("Ratio of Power Transmitted : Power Reflected");
    TGraph *gcheat = new TGraph(swrs.size());
    gcheat->SetTitle("Antenna magnitude, VSWR Corrected"); 

    for (int i = 0; i < gmag->GetN(); i++) {
      gswr->GetX()[i]   = freqs[i]; 
      gpr->GetX()[i]    = freqs[i]; 
      gmag->GetX()[i]   = freqs[i]; 
      gphase->GetX()[i] = freqs[i]; 
      gcheat->GetX()[i]  = freqs[i];

      gswr->GetY()[i] = swrs[i];

      double power_loss = 1.0 - ((swrs[i]-1.0)*(swrs[i]-1.0))/((swrs[i]+1.0)*(swrs[i]+1.0));
      gpr->GetY()[i] = power_loss;
      gcheat->GetY()[i] = mags[i] + 10.0 * TMath::Log10(power_loss);
      gmag->GetY()[i] = mags[i]; 
      gphase->GetY()[i] = phases[i]; 
    }

    double offset = gphase->Eval(0); 
    for (int i = 0; i < gphase->GetN(); i++) { gphase->GetY()[i]-=offset; }     

    // Save templates
    sprintf(graph_name,"phase_%i", int(theta));
    gphase->Write(graph_name);
    sprintf(graph_name,"cormag_%i", int(theta));
    gcheat->Write(graph_name);
    sprintf(graph_name,"mag_%i", int(theta));
    gmag->Write(graph_name);
    sprintf(graph_name,"swr_%i", int(theta));
    gswr->Write(graph_name);
    sprintf(graph_name,"pr_%i", int(theta));
    gpr->Write(graph_name);

    fclose(f); 
  }


  return 0; 
}
