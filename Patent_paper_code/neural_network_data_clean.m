T = readtable('../shot_logs.csv','Delimiter',',', ...
  'HeaderLines', 0, 'ReadVariableNames', true);

T.GAME_ID = [];
T.MATCHUP = [];
T.CLOSEST_DEFENDER = [];
T.CLOSEST_DEFENDER_PLAYER_ID = [];
T.FGM = [];
T.PTS = [];
T.player_name = [];
T.player_id = [];

T(isnan(T.SHOT_CLOCK), :) = [];

T.PERIOD = (T.PERIOD - 1).*(12*60);
T.GAME_CLOCK = arrayfun(@(x) calc_secs(char(x)), T.GAME_CLOCK);
T.GAME_TIME=T.PERIOD+T.GAME_CLOCK;
T.PERIOD = [];
T.GAME_CLOCK = [];


T.LOCATION = double(strcmp(T.LOCATION, 'A'));
T.LOCATION(T.LOCATION == 0) = -1;
T.W = double(strcmp(T.W, 'W'));
T.W(T.W == 0) = -1;
T.SHOT_RESULT = double(strcmp(T.SHOT_RESULT, 'made'));
%T.SHOT_RESULT(T.SHOT_RESULT == 0) = -1;


T = [T(:, 10) T(:, 1:9) T(:, 11:end)];%make shot result the first column
writetable(T, 'neural_network_data.csv');

function y = calc_secs(str)
    tokenized_string = strsplit(str, ':');
    y = 12*60-60*str2double(tokenized_string(1)) + str2double(tokenized_string(2));
end
