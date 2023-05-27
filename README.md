# CIAYN
## CCTV_Is_ALL_You_Need

-----------

KAIST 2023 Spring CS546 Wireless Mobile Internet

Authorized [Donghwan Kim](https://donghwankim0101.github.io/)

Authorized [Minje Kim](https://yunminjin2.github.io/)

-----------

# Table of Contents
1. [Getting Started]
        
    1.1. [Environments]
    
    1.2. [How to Test]

2. [Introduction]
3. [Method]
4. [Results]

## Getting Started

Start by cloning the repo:

        git clone https://github.com/yunminjin2/CIAYN.git
        cd CIAYN

### Environment

* Python >= 3.8

        pip install -r requirements.txt

### How to Test

We provide sample images, gps <-> pixel mapping and masking regions

* `--location`: location of CCTV image [GW, GS, KO, NC, SJ, SG, SB]
* `--index`: index of CCTV image [0, 1]
* `--compression`: compression methods [base(default), rectangle_crop, road_masking, sidewalk_masking]
* `--display`: use this for displaying the results

**Data size with naive method**

        python main.py --location GW --index 1 --display
        
**Data size with rectangle_crop method**

        python main.py --location GW --index 1 --compression rectangle_crop --display

**Data size with road_masking method**

        python main.py --location GW --index 1 --compression road_masking --display

**Data size with sidewalk_masking method**

        python main.py --location GW --index 1 --compression sidewalk_masking --display

## Introduction

## Method

## Results
