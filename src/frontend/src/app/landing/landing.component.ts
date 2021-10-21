import { Component, OnInit } from '@angular/core';
import { faCoffee } from '@fortawesome/free-solid-svg-icons';
import { ApiserviceService } from '../apiservice.service';


@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.css']
})
export class LandingComponent implements OnInit {
  dataset:any
  data = "";
  firstradio = true;
  imageSrc = "";
  imagegroupid = ""
  animalcount = 0
  animaltype = "";
  animalcheck = false;
  faCoffee = faCoffee;
  
  constructor(private service: ApiserviceService) { }
  
  datamapping(data): void {
    this.imageSrc = data.images[0];
    this.animalcount = data.animalcount;
    this.animaltype = data.animaltype
    this.imagegroupid = data.imagegroupid;
    this.dataset=data;
    console.log(this.animaltype)
  }


  ngOnInit(): void {
    this.service.getImages().subscribe((data:any)=>{
      this.datamapping(data)
    })
  }
  
  radioChange(event) {
    this.imageSrc = event;
  }

  submitdata(){
    this.data = "{'imagegroupid' :" + this.imagegroupid + ", 'animal' :" + this.animaltype + ", 'animalcount : " + this.animalcount + "}";

    this.service.postAnnotation(this.data).subscribe(data => {
      console.log(data)
      this.service.getImages().subscribe((data:any)=>{
        this.datamapping(data)
      });
    });
  }

  
}
