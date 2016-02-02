#!/usr/bin/perl
#
#  Author: Veselin Stoyanov
#  
#  Description: Aggregator for SemEval-2016 task 4, subtask D
#  Given individual tweet-level prediction, turns them into topic-level
#  aggregates. Used to generate the gold level data.
#
#  Last modified: Jan. 15, 2016
#


use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $INPUT_FILE         =  $ARGV[0];
my $OUTPUT_FILE        =  $INPUT_FILE . '.aggregate';


########################
###   MAIN PROGRAM   ###
########################

my %stats = ();

open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;

my $totalExamples = 0;
for (; <INPUT>; ) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### Check the input file format
	#1234	michael jordan positive
	die "Wrong file format!" if (!/^(\d+)\t([^\t]+)\t(positive|negative)/);
	my ($pid, $topic, $label) = ($1, $2, $3);

	$stats{$topic}{$label}++;
	$totalExamples++;
}

close(INPUT) or die;
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

### Initialize zero counts, compute probabilities and smooth
foreach my $topic (keys %stats) {
    my $topicCount = 0;
    foreach my $class ('positive', 'negative') {
	$stats{$topic}{$class} = 0 if (!defined($stats{$topic}{$class}));
	$topicCount += $stats{$topic}{$class};
    }
    #normalize
    print OUTPUT "$topic";
    foreach my $class (keys $stats{$topic}) {
	$stats{$topic}{$class} /= $topicCount;
	printf OUTPUT "\t$stats{$topic}{$class}";
    }
    printf OUTPUT "\n"
}


close(OUTPUT) or die;
