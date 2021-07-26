function status=GetSpectrogram(working_path,saving_path)
    load(working_path)
    if height(Calls)==0
        status=0
        return
    end

    for i = 1:height(Calls)

    one_call=Calls(i,:);

    fig=figure();
    set(fig,'visible','off');
    ax=axes();
    cla(ax);

    img=imagesc([],[],[],'Parent',ax);
    cb=colorbar(ax);
    cb.Label.String = 'Amplitude';
    cb.Color = [0 0 0];
    cb.FontSize = 12;
    ylabel(ax,'Frequency (kHz)','Color','w');
    xlabel(ax,'Time (s)','Color','w');
    set(ax,'Color',[0 0 0]);
    colormap(ax,'inferno');
    bo=rectangle('Position',[1 1 1 1],'Curvature',0.2,'EdgeColor','g',...
        'LineWidth',3,'Parent', ax);

    [I,windowsize,noverlap,nfft,rate,box,s,fr,ti,audio,AudioRange]=CreateSpectrogram(one_call);
    set(ax,'YDir', 'normal','YColor',[0 0 0],'XColor',[0 0 0],'Clim',[0 2*mean(max(I))]);
    set(img,'CData',imgaussfilt(abs(s)),'XData',ti,'YData',fr/1000);
    set(ax,'Xlim',[img.XData(1) img.XData(end)])
    set(ax,'ylim',[40 90]);

    set(bo,'Position',one_call.RelBox(1, :),'EdgeColor','g');

    sp=[saving_path,'\\spectrogram',num2str(i),'.png'];
    saveas(fig,sp);
    sp=[saving_path,'\\spectrogram',num2str(i),'.mat'];
    spect=struct('I',I,'windowsize',windowsize,'noverlap',noverlap,'nfft',nfft,'rate',rate,'box',box,'s',s,'fr',fr,'ti',ti,'audio',audio,'AudioRange',AudioRange);
    save(sp,'spect');
    status=1
    end
