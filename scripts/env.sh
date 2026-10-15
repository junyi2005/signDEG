# Paths used by every script in this folder. Edit, then run the scripts from the repository root.
export W=${W:-weights}                     # dinov3-vitb16/, yolo/, byt5-base/ (README, Section 2)
export T=${T:-third_party}                 # WiLoR code release (with its checkpoints and MANO files)
export D=${D:-/path/to/datasets}           # corpora (README, Section 3)
export O=${O:-/path/to/work}               # lists/, parts/, stores/, runs/
export NGPU=${NGPU:-4}                     # GPUs per node for pre-training and translation
export YTASL_TABLE=${YTASL_TABLE:-$D/youtube_asl/youtube_asl.tsv}   # caption table of all source videos
export SOURCE_VIDEOS=${SOURCE_VIDEOS:-$D/youtube_asl/videos,$D/youtube_sl25/videos}
export OPENASL_TABLE=${OPENASL_TABLE:-$D/openasl/openasl-v1.0.tsv}
export OPENASL_VIDEOS=${OPENASL_VIDEOS:-$D/openasl/videos}
export YTSL25_LIST=${YTSL25_LIST:-$D/youtube_sl25/asl_video_ids.txt}
export H2S=${H2S:-$D/how2sign}                     # how2sign_{train,val,test}.csv and clips/<SENTENCE_NAME>.mp4
export ASLC=${ASLC:-$D/ASL_Citizen}                # videos/, splits/
export PYOPENGL_PLATFORM=egl
export R=$O/runs/ssl                               # pre-training checkpoints (signdeg_{lh,rh,face})
DET="--dinov3 $W/dinov3-vitb16 --face-ckpt $W/yolo/yolov8n-face.pt --hand-ckpt $W/yolo/hand_yolov8n.pt"
UP="$R/signdeg_lh/ckpt_final.pt $R/signdeg_rh/ckpt_final.pt $R/signdeg_face/ckpt_final.pt"
