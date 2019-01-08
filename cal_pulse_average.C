#pragma cling load("/home/danielsmith/nuphase/lib/libnuphaseroot.so");
#pragma cling load("/home/danielsmith/RootFftwInstall/lib/libRootFftwWrapper.so.3.0.1");

#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include "nuphaseEvent.h" 
#include "nuphaseTools.h" 
#include "nuphaseHeader.h" 
#include "nuphaseTools.h"

#include "AnalyticSignal.h" 
#include "FFTtools.h" 

// To run this script:
// $ root -l "cal_pulse_average.C(\"/project2/avieregg/nuphase/telem/root/run437\")"

// Different used runs:
// run376 : held at 127 m
// run381 : held at 153 m
// run437 : Broadside cal pulser run

// Drop runs require different averaging script
// run379 : first half of drop
// run391 : second half of drop

void cal_pulse_average(char* path) { 

  bool bVerbose = false;
  bool bPlot = false;
  int iMaxNGraphs = 10000;
  int iNChans = 8;
  float fThreshHold = 45.0;
  float fBaseline = 63.15569564;
  UInt_t iMinTime = 1516350110;
  UInt_t iMaxTime = 1516352800;

  // Histograms
  TH2F* hPtoPvsReadoutTime = new TH2F("hPtoPvsReadoutTime", "hPtoPvsReadoutTime", 200, iMinTime, iMaxTime, 150, 0.0, 150.0);

  //UInt_t min_time_cut = 1516351110;
  //UInt_t max_time_cut = 1516351800;

  // Loading the event file
  std::ostringstream event_string;
  std::ostringstream head_string;
  event_string << path << "/event.root";
  head_string << path << "/header.filtered.root";
  ifstream ftest0(event_string.str().c_str());
  ifstream ftest1(head_string.str().c_str());
  if(!ftest0.good() or !ftest1.good()) { if(bVerbose) { std::cout << "Bad Path. " << std::endl; } return; } 
  TFile *rEvent  = new TFile(event_string.str().c_str());
  TFile *rHeader = new TFile(head_string.str().c_str());
  TTree *rTreeEvent;
  TTree *rTreeHeader;
  rEvent->GetObject("event", rTreeEvent);
  rHeader->GetObject("header", rTreeHeader);
  nuphase::Event* evt = new nuphase::Event();
  nuphase::Header* head = new nuphase::Header();
  rTreeEvent->SetBranchAddress("event", &evt); // load tree into this structure.
  rTreeHeader->SetBranchAddress("header", &head); // load tree into this structure.
  
  TGraph* gGraphsToAvg[iNChans][iMaxNGraphs];
  int iGraphsToAvg = 0;

  Long64_t nentries = rTreeEvent->GetEntriesFast();  
  for(size_t jentry = 0; jentry < nentries; jentry++) {
    rTreeEvent->GetEntry(jentry);
    rTreeHeader->GetEntry(jentry);

    if(evt->getEventNumber() != head->event_number) { 
      if(bVerbose) { std::cout << "Header does not match event. " << std::endl; }
      break;
    }

    // RF trigger
    if(head->trigger_type != 2) { continue; }

    // Trigger time aligned with expected time
    // This step could be better, using corrected time instead of readout time, 
    // but works reasonable well. 
    if(double(head->readout_time_ns[0])*1e-9 < 0.99) { continue; }

    // peak-to-peak difference for max and min, pretending like one pulse
    bool bFoundPulse = false;
    for(size_t iChan = 0; iChan < iNChans; iChan++) { 
      if(iChan == 5) { continue; } // chan5 broken

      double fPk2Pk = nuphase::Tools::getPKtoPK(evt->getGraph(iChan));
      hPtoPvsReadoutTime->Fill(head->readout_time[0], fPk2Pk);

      if(fPk2Pk > fThreshHold) { bFoundPulse = true; }
    }

    if(!bFoundPulse) { continue; }
    
    for(size_t iChan = 0; iChan < iNChans; iChan++) { 
      gGraphsToAvg[iChan][iGraphsToAvg] = evt->getGraph(iChan);

      if(bVerbose) { std::cout << "NSamples = " << evt->getGraph(iChan)->GetN() << std::endl; } 
      
      for(size_t i = 0; i < gGraphsToAvg[iChan][iGraphsToAvg]->GetN(); i++) { 	
	// by default, nuphaseroot changes TDCs to ns. Change back so extrapolation late is easier. 
	gGraphsToAvg[iChan][iGraphsToAvg]->GetX()[i] *= 3.0 / 2.0;

	// Subtract baseline.
	gGraphsToAvg[iChan][iGraphsToAvg]->GetY()[i] -= fBaseline;
      }
    } 
    iGraphsToAvg += 1;

  } // end of jentries

  if(bVerbose) { std::cout << " End of jEntries. nGraphs = " << iGraphsToAvg << std::endl; }

  // If no graphs found, break
  if(iGraphsToAvg < 2) { 
    if(bVerbose) { std::cout << "Too few graphs to correlate and average. " << std::endl; }
    return; 
  }

  // Correlate and average
  TGraph* gChanAvg[iNChans]; 
  for(size_t iChan = 0; iChan < iNChans; iChan++) {
    TGraph* gTempGraph[iGraphsToAvg];
    
    // First, interpolate
    for(size_t i = 0; i < iGraphsToAvg; i++) {
      gTempGraph[i] = FFTtools::getInterpolatedGraphFreqDom(gGraphsToAvg[iChan][i], 0.01);
    }

    // then average
    gChanAvg[iChan] = FFTtools::correlateAndAverage(iGraphsToAvg, gTempGraph);

    if(bVerbose) { std::cout << "Done with chan" << iChan << std::endl; }
  }

  // Save templates
  TFile* f = new TFile("cal_pulse_templates.root", "RECREATE");

  TGraph* gGoodChans[iNChans-1];
  char *cGraphName = new char[10];
  int iRun = 0;
  for(size_t i = 0; i < 8; i++) {
    sprintf(cGraphName,"template_%i",int(i));
    gChanAvg[i]->Write(cGraphName);

    if(i != 5) {
      gGoodChans[iRun] = gChanAvg[i];
      iRun += 1;
    }
  }

  // Finally, correlate all the templates and save
  TGraph* gFinal = FFTtools::correlateAndAverage(iNChans-1, gGoodChans);
  gFinal->Write("template");

  if(bPlot) { 
    // Plot each channel's waveform
    std::vector<int> my_colors = {-10, -9, -7, -4, 0, 1, 2, 4};
    TCanvas *c1 = new TCanvas ("c1","Each Channel",1);
    gStyle->SetOptStat(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetOptTitle(0);
    c1->Divide(1,8);
    for(size_t i = 0; i < iNChans; i++) {
      c1->cd(i+1);     
      
      gPad->SetTopMargin(0.0);
      gPad->SetRightMargin(0.01);
      if(i != iNChans-1) { gPad->SetBottomMargin(0.05); }

      gChanAvg[i]->SetTitle("");
      gChanAvg[i]->Draw();
      gChanAvg[i]->GetYaxis()->SetLabelSize(0.1);
    }
    c1->Update();

    // Plot gFinal full template
    TCanvas *c2 = new TCanvas ("c2","All Channels Correlated",1);
    gFinal->GetXaxis()->SetTitle("Time [ns]");
    gFinal->GetYaxis()->SetTitle("Voltage [abu]");
    gFinal->Draw();
    c2->Update();

    // Plot pk2pk vs readout_time
    TCanvas *c3 = new TCanvas ("c3","Pk-2-Pk vs. Time",1);
    hPtoPvsReadoutTime->GetXaxis()->SetTimeDisplay(1);
    hPtoPvsReadoutTime->GetXaxis()->SetTimeFormat("#splitline{%d-%m-%y}{%H:%M}");
    hPtoPvsReadoutTime->GetXaxis()->SetTimeOffset(0,"gmt");
    hPtoPvsReadoutTime->GetXaxis()->SetLabelOffset(0.012);
    hPtoPvsReadoutTime->GetXaxis()->SetTitleOffset(1.2);
    hPtoPvsReadoutTime->GetXaxis()->SetNdivisions(8); 
    hPtoPvsReadoutTime->GetXaxis()->SetTitle("Readout time in UTC [DD/MM/YY]");
    hPtoPvsReadoutTime->GetYaxis()->SetTitle("Peak-to-peak Difference [abu]");
    hPtoPvsReadoutTime->Draw("colz");
    c3->Update();
  }

}
