from pydub import AudioSegment
import glob
import os
path = '/home/momozyc/Documents/LMS_codes/speech-music-classify-Luo/mp3'
export_path = os.path.join(path, 'music/')
path1 = '/home/momozyc/Documents/LMS_codes/speech-music-classify-Luo'

'''
music_mp3_files = glob.glob(os.path.join(path, '*.mp3'))
i = 0
for mp3 in music_mp3_files:
    print(mp3)
    sound = AudioSegment.from_mp3(mp3)
    name = i
    print(path+str(name)+'.wav')
    sound.export(path+str(name)+'.wav',format ='wav')
    i += 1
'''
music_wav_files = glob.glob(os.path.join(path1, '*.wav'))
#print(music_wav_files)
i = 76
for wav in music_wav_files:
    #print(wav)
    sound = AudioSegment.from_file(wav)
    name = i
    sound = sound.set_frame_rate(16000).set_channels(1)
    print(sound.frame_rate)
    sound.export(path1+'/music/'+str(name)+'.wav',format ='wav')
    i += 1
